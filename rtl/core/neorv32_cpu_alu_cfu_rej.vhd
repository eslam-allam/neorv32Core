-- ================================================================================ --
-- NEORV32 CPU - ALU Custom Functions Unit (CFU) - rejection-sampling engine       --
-- -------------------------------------------------------------------------------- --
-- ML-KEM's uniform rejection sampling (FIPS 203, Algorithm 13 "SampleNTT"):        --
--                                                                                  --
--   while (ctr < target && pos + 3 <= buflen) {                                    --
--     val0 = ((buf[pos+0] >> 0) | (buf[pos+1] << 8)) & 0xFFF;                       --
--     val1 = ((buf[pos+1] >> 4) | (buf[pos+2] << 4)) & 0xFFF;                       --
--     pos += 3;                                                                    --
--     if (val0 < MLKEM_Q) { r[ctr++] = val0; }                                     --
--     if (ctr < target && val1 < MLKEM_Q) { r[ctr++] = val1; }                     --
--   }                                                                              --
-- i.e. THREE BYTES -> TWO 12-bit candidates, the second window straddling the byte  --
-- boundary, each accepted with probability q/4096 = 3329/4096 = 81.3 %.            --
--                                                                                  --
-- WHY A SEPARATE ENGINE and not an op in the NTT engine: the kernel is a BYTE-     --
-- STREAM FILTER, not a polynomial operation. The NTT engine has a buffer (its      --
-- scratchpad) and a comparator, but neither the extractor nor the compaction - the --
-- datapath has to be built either way - and the engine that carries the NTT        --
-- (38-70x) and the three arithmetic ops (18.8x / 5.1x / 3.5x) is the wrong place   --
-- to add a byte-stream filter. The design study is in neorv32.sw/docs/cfu_arith.md, --
-- section 3g; the measured prize is 195,858 cycles per keygen (27.2 % of the       --
-- matrix expansion).                                                               --
--                                                                                  --
-- DATAPATH: a byte-level shift register, so the 3-byte stride against the 4-byte  --
-- CFU port needs no alignment muxing at all - the bytes are fed in one at a time   --
-- into a 24-bit group register, and a group is complete exactly when its third    --
-- byte arrives. One PUSH op consumes one 32-bit word (four bytes) in four cycles.  --
-- The alternative (extracting a 24-bit group at a byte offset that cycles through  --
-- six positions, because 3 and 4 are coprime) would need a 6-way 24-bit mux and a   --
-- 96-bit window.                                                                   --
--                                                                                  --
-- The output is COMPACTED in hardware (only the accepted candidates are kept) and  --
-- packed two per 32-bit word, because the coefficients have to land in the        --
-- library's int16_t array and a half-word store is expensive on this core.        --
--                                                                                  --
-- The NEORV32 RISC-V Processor - https://github.com/stnolting/neorv32                --
-- Copyright (c) NEORV32 contributors.                                                --
-- Copyright (c) 2020 - 2026 Stephan Nolting. All rights reserved.                    --
-- Licensed under the BSD-3-Clause license, see LICENSE for details.                  --
-- SPDX-License-Identifier: BSD-3-Clause                                              --
-- ================================================================================ --

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neorv32_cpu_alu_cfu_rej is
  port (
    -- global control --
    clk_i    : in  std_ulogic; -- global clock, rising edge
    rstn_i   : in  std_ulogic; -- global reset, low-active, async
    -- request --
    start_i  : in  std_ulogic; -- start trigger, single-shot
    inst_i   : in  std_ulogic_vector(31 downto 0); -- full instruction word
    rs1_i    : in  std_ulogic_vector(31 downto 0); -- register source operand 1
    rs2_i    : in  std_ulogic_vector(31 downto 0); -- register source operand 2
    -- response --
    result_o : out std_ulogic_vector(31 downto 0); -- operation result
    valid_o  : out std_ulogic                      -- operation done; result valid
  );
end neorv32_cpu_alu_cfu_rej;

architecture neorv32_cpu_alu_cfu_rej_rtl of neorv32_cpu_alu_cfu_rej is

  -- instruction decode --
  constant opcode_custom0_c : std_ulogic_vector(6 downto 0) := "0001011"; -- CUSTOM-0 opcode
  constant funct3_rej_c     : std_ulogic_vector(2 downto 0) := "111";     -- rejection engine

  -- sub-opcodes (funct7) --
  constant sub_setup_c : std_ulogic_vector(6 downto 0) := "0000000"; -- reset + target/limit
  constant sub_push_c  : std_ulogic_vector(6 downto 0) := "0000001"; -- feed one input word
  constant sub_drain_c : std_ulogic_vector(6 downto 0) := "0000010"; -- pop one output word

  -- ML-KEM's modulus: a candidate is accepted iff it is < q --
  constant kem_q_c : integer := 3329;

  -- instruction decode (combinational) --
  signal op_valid : std_ulogic;
  signal start    : std_ulogic;
  signal subop    : std_ulogic_vector(6 downto 0);

  -- sequencer --
  type state_t is (S_IDLE, S_BYTE, S_DONE);
  signal state : state_t;

  -- input side: the word being consumed and the group being assembled --
  signal wreg   : std_ulogic_vector(31 downto 0); -- the current input word
  signal wcount : unsigned(2 downto 0);            -- bytes left in the word (0..4)
  signal gbuf   : std_ulogic_vector(23 downto 0); -- the group's 3 bytes (b0,b1,b2)
  signal gcount : unsigned(1 downto 0);            -- bytes of the group so far (0..2)

  -- counters --
  signal gcnt   : unsigned(15 downto 0); -- groups processed
  signal glimit : unsigned(15 downto 0); -- the maximum number of groups
  signal cnt    : unsigned(15 downto 0); -- coefficients produced
  signal target : unsigned(15 downto 0); -- the coefficient target

  -- output side: a two-word buffer plus the pending half-word --
  signal obuf0  : std_ulogic_vector(31 downto 0); -- output word 0
  signal obuf1  : std_ulogic_vector(31 downto 0); -- output word 1
  signal ocount : unsigned(1 downto 0);            -- words in the buffer (0..2)
  signal acc    : std_ulogic_vector(15 downto 0); -- the pending coefficient
  signal accful : std_ulogic;                     -- the acc holds a coefficient

  -- combinational datapath --
  signal byte_i    : std_ulogic_vector(7 downto 0);  -- the byte being fed
  signal gbuf_next : std_ulogic_vector(23 downto 0); -- the group with that byte
  signal val0      : unsigned(11 downto 0);         -- the first candidate
  signal val1      : unsigned(11 downto 0);         -- the second candidate
  signal acc0      : std_ulogic;                   -- candidate 0 accepted
  signal acc1      : std_ulogic;                   -- candidate 1 accepted
  signal inc0      : unsigned(0 downto 0);          -- the count increment of candidate 0
  signal inc1      : unsigned(0 downto 0);          -- the count increment of candidate 1
  signal ncnt      : unsigned(15 downto 0);         -- the new coefficient count

  -- the group may only be consumed if there is data, the group limit allows it
  -- and the coefficient target has not been reached --
  signal grp_ok : std_ulogic;

  -- the output word the DRAIN op returns --
  signal drain_val : std_ulogic_vector(31 downto 0);

begin

  -- Instruction Decode ---------------------------------------------------------------------
  op_valid <= '1' when (inst_i(6 downto 0) = opcode_custom0_c) and (inst_i(14 downto 12) = funct3_rej_c) else '0';
  start    <= start_i when (op_valid = '1') else '0';
  subop    <= inst_i(31 downto 25);

  -- Input Datapath ------------------------------------------------------------------------
  -- The byte is the low half of the word register; the register is shifted right by one
  -- byte per cycle, so the byte comes out for free.
  --
  -- The group register holds the group's THREE BYTES in the library's order: b0 in the low
  -- bits, b1 in the middle, b2 in the high bits. The bytes are therefore shifted in from
  -- the TOP (gbuf >> 8 | byte << 16), so that after three bytes the register holds
  -- b0 | b1<<8 | b2<<16 - exactly the library's 24-bit group.
  byte_i    <= wreg(7 downto 0);
  gbuf_next <= std_ulogic_vector(shift_right(unsigned(gbuf), 8)) or
               std_ulogic_vector(shift_left(resize(unsigned(byte_i), 24), 16));
  grp_ok    <= '1' when ((gcnt < glimit) and (cnt < target)) else '0';

  -- The two candidates of the group. The library's group is b0 | b1<<8 | b2<<16, so
  --   val0 = (b0 | b1<<8) & 0xFFF   = the group's bits [11:0]
  --   val1 = ((b1>>4) | (b2<<4))     = the group's bits [23:12]
  -- (both are 12-bit windows of the byte stream, at the bit offsets 0 and 12).
  val0 <= unsigned(gbuf_next(11 downto 0));
  val1 <= unsigned(gbuf_next(23 downto 12));

  -- The acceptance test and the count. The library's second candidate is only taken when
  -- the count is still below the target AFTER the first one was taken. The group limit
  -- gates both: a group beyond the limit (the tail of the buffer) must not be consumed.
  -- (The increments are separate signals rather than conditional expressions inside the
  -- arithmetic, because the design is read as VHDL-93.)
  acc0 <= '1' when (val0 < to_unsigned(kem_q_c, 12)) and (grp_ok = '1') else '0';
  acc1 <= '1' when (val1 < to_unsigned(kem_q_c, 12)) and (grp_ok = '1') and
                   ((cnt + resize(inc0, 16)) < target) else '0';
  inc0 <= (others => '0') when (acc0 = '0') else (others => '1');
  inc1 <= (others => '0') when (acc1 = '0') else (others => '1');
  ncnt <= cnt + resize(inc0, 16) + resize(inc1, 16);

  -- Output Datapath -----------------------------------------------------------------------
  -- The compacted coefficients are packed two per word, the older one in the low half.
  -- The DRAIN op returns the buffer's first word, or the pending half-word (in the low
  -- half) when the buffer is empty.
  drain_val <= obuf0 when (ocount /= 0) else
               std_ulogic_vector(resize(unsigned(acc), 32));

  -- Sequencer -------------------------------------------------------------------------------
  seq_proc: process(rstn_i, clk_i)
    -- the output compaction, as a function of the current state and the new coefficients --
    variable v_cnt   : unsigned(1 downto 0);
    variable v_ocnt  : unsigned(1 downto 0);
    variable v_accf  : std_ulogic;
    variable v_acc   : std_ulogic_vector(15 downto 0);
    variable v_obuf0 : std_ulogic_vector(31 downto 0);
    variable v_obuf1 : std_ulogic_vector(31 downto 0);
    variable v_wreg  : std_ulogic_vector(31 downto 0);
    variable v_wcnt  : unsigned(2 downto 0);
    variable v_gbuf  : std_ulogic_vector(23 downto 0);
    variable v_gcnt  : unsigned(1 downto 0);
    variable v_gcnt2 : unsigned(15 downto 0);
    variable v_cntn  : unsigned(15 downto 0);
    variable c0, c1  : std_ulogic_vector(15 downto 0);
  begin
    if (rstn_i = '0') then
      state  <= S_IDLE;
      wreg   <= (others => '0');
      wcount <= (others => '0');
      gbuf   <= (others => '0');
      gcount <= (others => '0');
      gcnt   <= (others => '0');
      glimit <= (others => '0');
      cnt    <= (others => '0');
      target <= (others => '0');
      obuf0  <= (others => '0');
      obuf1  <= (others => '0');
      ocount <= (others => '0');
      acc    <= (others => '0');
      accful <= '0';
    elsif rising_edge(clk_i) then
      case state is

        when S_IDLE =>
          if (start = '1') then
            case subop is
              -- SETUP: rs1 = {the group limit (15:0), the coefficient target (15:0)} --
              when sub_setup_c =>
                wreg   <= (others => '0');
                wcount <= (others => '0');
                gbuf   <= (others => '0');
                gcount <= (others => '0');
                gcnt   <= (others => '0');
                cnt    <= (others => '0');
                glimit <= unsigned(rs1_i(31 downto 16));
                target <= unsigned(rs1_i(15 downto 0));
                obuf0  <= (others => '0');
                obuf1  <= (others => '0');
                ocount <= (others => '0');
                acc    <= (others => '0');
                accful <= '0';
              -- PUSH: rs1 = the next input word (four bytes) --
              when sub_push_c =>
                wreg   <= rs1_i;
                wcount <= to_unsigned(4, 3);
                state  <= S_BYTE;
              -- DRAIN: pop one output word --
              when sub_drain_c =>
                if (ocount = 2) then
                  obuf0  <= obuf1;
                  obuf1  <= (others => '0');
                  ocount <= ocount - 1;
                elsif (ocount = 1) then
                  obuf0  <= (others => '0');
                  ocount <= (others => '0');
                else
                  accful <= '0'; -- the pending half-word was handed out
                end if;
              when others =>
                null;
            end case;
          end if;

        -- Feed the word's bytes into the group register, one per cycle --
        when S_BYTE =>
          if (wcount > 0) then
            v_wreg := std_ulogic_vector(shift_right(unsigned(wreg), 8));
            v_wcnt := wcount - 1;
            v_gbuf := std_ulogic_vector(shift_right(unsigned(gbuf), 8)) or
                      std_ulogic_vector(shift_left(resize(unsigned(byte_i), 24), 16));
            v_gcnt := gcount;
            if (gcount = 2) then
              -- the group is complete: test and compact its two candidates --
              v_gcnt  := to_unsigned(0, 2);
              v_gcnt2 := gcnt + 1;
              v_cntn  := ncnt;
              c0 := std_ulogic_vector(resize(unsigned(val0), 16));
              c1 := std_ulogic_vector(resize(unsigned(val1), 16));
              v_cnt  := ocount;
              v_ocnt := ocount;
              v_accf := accful;
              v_acc  := acc;
              v_obuf0 := obuf0;
              v_obuf1 := obuf1;
              if (grp_ok = '1') then
                if (acc0 = '1') then
                  if (v_accf = '1') then
                    -- the acc + candidate 0 form a word; candidate 1 becomes the new acc
                    if (v_ocnt = 0) then
                      v_obuf0 := std_ulogic_vector(c0) & v_acc;
                    else
                      v_obuf1 := std_ulogic_vector(c0) & v_acc;
                    end if;
                    v_ocnt := v_ocnt + 1;
                    v_accf := '1';
                    v_acc  := c1;
                    if (acc1 = '0') then
                      v_accf := '0';
                    end if;
                  else
                    -- the acc is empty: candidate 0 starts the next word
                    v_acc  := c0;
                    v_accf := '1';
                    if (acc1 = '1') then
                      -- candidate 1 completes that word
                      if (v_ocnt = 0) then
                        v_obuf0 := std_ulogic_vector(c1) & c0;
                      else
                        v_obuf1 := std_ulogic_vector(c1) & c0;
                      end if;
                      v_ocnt := v_ocnt + 1;
                      v_accf := '0';
                    end if;
                  end if;
                elsif (acc1 = '1') then
                  -- only candidate 1 is accepted
                  if (v_accf = '1') then
                    if (v_ocnt = 0) then
                      v_obuf0 := std_ulogic_vector(c1) & v_acc;
                    else
                      v_obuf1 := std_ulogic_vector(c1) & v_acc;
                    end if;
                    v_ocnt := v_ocnt + 1;
                    v_accf := '0';
                  else
                    v_acc  := c1;
                    v_accf := '1';
                  end if;
                end if;
              end if;
              wreg   <= std_ulogic_vector(v_wreg);
              wcount <= v_wcnt;
              gbuf   <= v_gbuf;
              gcount <= v_gcnt;
              gcnt   <= v_gcnt2;
              cnt    <= v_cntn;
              obuf0  <= v_obuf0;
              obuf1  <= v_obuf1;
              ocount <= v_ocnt;
              acc    <= v_acc;
              accful <= v_accf;
              if (v_wcnt = 0) then
                state <= S_DONE; -- the word's bytes are consumed
              end if;
            else
              wreg   <= std_ulogic_vector(v_wreg);
              wcount <= v_wcnt;
              gbuf   <= v_gbuf;
              gcount <= v_gcnt + 1;
            end if;
          else
            state <= S_DONE; -- no more bytes: a partial group is discarded
          end if;

        when S_DONE =>
          state <= S_IDLE;

        when others =>
          state <= S_IDLE;

      end case;
    end if;
  end process seq_proc;

  -- Function Result Select ------------------------------------------------------------------
  -- PUSH returns the number of coefficients produced so far (valid when the op ends);
  -- DRAIN returns the next output word; SETUP returns zero.
  result_o <= drain_val when (subop = sub_drain_c) else
               std_ulogic_vector(resize(cnt, 32));

  -- NOTE: the design is read as VHDL-93, where a conditional EXPRESSION inside a
  -- process is not legal (it needs VHDL-2008). So the "if/else" form is used here
  -- rather than the "when/else" expression - the same rule the NTT engine follows.
  valid_proc: process(op_valid, state, subop, start)
  begin
    valid_o <= '0';
    if (op_valid = '1') then
      case subop is
        when sub_setup_c | sub_drain_c =>
          if (state = S_IDLE) and (start = '1') then
            valid_o <= '1';
          else
            valid_o <= '0';
          end if;
        when sub_push_c =>
          if (state = S_DONE) then
            valid_o <= '1';
          else
            valid_o <= '0';
          end if;
        when others =>
          valid_o <= '0';
      end case;
    end if;
  end process valid_proc;

end neorv32_cpu_alu_cfu_rej_rtl;
