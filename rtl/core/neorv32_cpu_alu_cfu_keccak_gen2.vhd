-- ================================================================================ --
-- NEORV32 PQC - Keccak-f1600 CFU engine (gen 2: 16-bit serialized datapath)
-- -------------------------------------------------------------------------------- --
-- The area-optimized Keccak-f1600 engine: a narrow (16-bit) serialized round
-- datapath with the 1600-bit state in distributed RAM instead of flip-flops.
-- A PERM op advances exactly ONE round (~487 cycles), so a permutation is 24 ops;
-- this keeps every co-processor op inside the NEORV32 CFU execution window
-- (alu_cp_tmo_c, 512 cycles) without touching the CPU core.
--
-- See neorv32_cpu_alu_cfu_keccak.vhd for the entity and the gen-1 counterpart.
--
-- The NEORV32 RISC-V Processor - https://github.com/stnolting/neorv32                --
-- Copyright (c) NEORV32 contributors.                                                --
-- Copyright (c) 2020 - 2026 Stephan Nolting. All rights reserved.                    --
-- Licensed under the BSD-3-Clause license, see LICENSE for details.                  --
-- SPDX-License-Identifier: BSD-3-Clause                                              --
-- ================================================================================ --

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library neorv32;
use neorv32.neorv32_package.all;

architecture neorv32_cpu_alu_cfu_keccak_gen2 of neorv32_cpu_alu_cfu_keccak is
  -- instruction decode --
  constant opcode_custom0_c : std_ulogic_vector(6 downto 0) := "0001011"; -- CUSTOM-0 opcode
  constant last_round_c     : integer := 23; -- index of the final Keccak-f1600 round
  constant cfu_write_lo_c   : std_ulogic_vector(2 downto 0) := "000";
  constant cfu_write_hi_c   : std_ulogic_vector(2 downto 0) := "001";
  constant cfu_read_lo_c    : std_ulogic_vector(2 downto 0) := "010";
  constant cfu_read_hi_c    : std_ulogic_vector(2 downto 0) := "011";
  constant cfu_perm_c       : std_ulogic_vector(2 downto 0) := "100";
  constant cfu_clear_c      : std_ulogic_vector(2 downto 0) := "101";

  -- rho rotation offsets rho[x][y] --
  type rho_array_t is array (0 to 4, 0 to 4) of natural;
  constant rho_c : rho_array_t := (
    ( 0, 36,  3, 41, 18),
    ( 1, 44, 10, 45,  2),
    (62,  6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39,  8, 14)
  );

  -- iota round constants --
  type rc_array_t is array (0 to 23) of std_ulogic_vector(63 downto 0);
  constant rc_c : rc_array_t := (
    x"0000000000000001", x"0000000000008082", x"800000000000808a", x"8000000080008000",
    x"000000000000808b", x"0000000080000001", x"8000000080008081", x"8000000000008009",
    x"000000000000008a", x"0000000000000088", x"0000000080008009", x"000000008000000a",
    x"000000008000808b", x"800000000000008b", x"8000000000008089", x"8000000000008003",
    x"8000000000008002", x"8000000000000080", x"000000000000800a", x"800000008000000a",
    x"8000000080008081", x"8000000000008080", x"0000000080000001", x"8000000080008008"
  );

  -- rho: chunk-scale part (number of whole 16-bit chunks to permute) --
  function rho_q(x, y : integer) return integer is
  begin
    return rho_c(x, y) / 16;
  end function;

  -- rho: sub-chunk shift --
  function rho_s(x, y : integer) return integer is
  begin
    return rho_c(x, y) mod 16;
  end function;

  -- pi: destination lane of source lane (x,y): lane = y + 5*((2x+3y) mod 5) --
  function pi_dst(x, y : integer) return integer is
  begin
    return y + 5*((2*x + 3*y) mod 5);
  end function;

  -- extract 16-bit chunk i (mod 4) of a 64-bit lane value --
  function chunk_of(v : std_ulogic_vector(63 downto 0); i : integer) return std_ulogic_vector is
  begin
    case (i mod 4) is
      when 0      => return v(15 downto 0);
      when 1      => return v(31 downto 16);
      when 2      => return v(47 downto 32);
      when others => return v(63 downto 48);
    end case;
  end function;

  -- funnel shift: out = (a<<s) | (b>>(16-s)); s=0 -> out = a --
  function funnel(a, b : std_ulogic_vector(15 downto 0); s : integer) return std_ulogic_vector is
    variable ua, ub : unsigned(15 downto 0);
    variable t      : unsigned(31 downto 0);
  begin
    ua := unsigned(a);
    ub := unsigned(b);
    t  := shift_right(ua & ub, 16 - s); -- {a,b}: a is the high half
    return std_ulogic_vector(t(15 downto 0));
  end function;

  -- rotate left by 1 --
  function rol1(v : std_ulogic_vector(63 downto 0)) return std_ulogic_vector is
  begin
    return v(62 downto 0) & v(63 downto 63);
  end function;

  -- single bit to integer --
  function b2i(b : std_ulogic) return integer is
  begin
    if (b = '1') then
      return 1;
    else
      return 0;
    end if;
  end function;

  -- state banks: 64 entries x 32 bit (only 0..49 used); entry 2*L+h = lane L half h --
  type bank_t is array (0 to 63) of std_ulogic_vector(31 downto 0);
  -- C[]: 5 columns x 64 bit, as 16 x 32-bit entries (entry 2*x+h) --
  type cram_t is array (0 to 15) of std_ulogic_vector(31 downto 0);

  signal bank_a, bank_b : bank_t := (others => (others => '0')); -- A = canonical, B = scratch
  signal c_ram          : cram_t := (others => (others => '0'));

  -- read ports (asynchronous; mirrored by the engine and by the CPU interface) --
  signal addr_a, addr_b : std_ulogic_vector(5 downto 0);
  signal addr_c         : std_ulogic_vector(3 downto 0);
  signal rd_a, rd_b, rd_c : std_ulogic_vector(31 downto 0);

  -- instruction decode --
  signal op_valid : std_ulogic;
  signal start    : std_ulogic;
  signal funct3   : std_ulogic_vector(2 downto 0);
  signal lane_idx : std_ulogic_vector(4 downto 0);
  signal lane_ok  : std_ulogic;

  -- the engine's response is the entity's result_o / valid_o port

  -- sequencer --
  type state_t is (S_IDLE, S_CLEAR, S_THA, S_THW, S_DLD, S_DCM,
                   S_RPLD, S_RPROT, S_RPWR, S_CHLD, S_CHWR, S_RND);
  signal state : state_t;
  signal rnd   : unsigned(4 downto 0);
  signal col   : unsigned(2 downto 0); -- x
  signal row   : unsigned(2 downto 0); -- y
  signal sub   : unsigned(5 downto 0);
  signal hbit  : std_ulogic;

  -- datapath registers --
  signal acl, ach   : std_ulogic_vector(31 downto 0); -- theta column accumulators
  signal v64, d64   : std_ulogic_vector(63 downto 0);
  signal cl64, ch64 : std_ulogic_vector(63 downto 0);
  signal rot64      : std_ulogic_vector(63 downto 0);
  signal r0, r1, r2 : std_ulogic_vector(31 downto 0);
  signal chi_val    : std_ulogic_vector(31 downto 0);
  signal done       : std_ulogic;

begin

  -- Instruction Decode ---------------------------------------------------------------------
  op_valid <= '1' when (inst_i(6 downto 0) = opcode_custom0_c) else '0';
  start    <= start_i when (op_valid = '1') else '0';
  funct3   <= inst_i(14 downto 12);
  lane_idx <= rs2_i(4 downto 0);
  lane_ok  <= '1' when (unsigned(rs2_i) < 25) else '0';

  -- State Banks (distributed RAM inference; no reset on the arrays!) -----------------------
  bank_a_proc: process(clk_i)
  begin
    if rising_edge(clk_i) then
      if (state = S_CHWR) then
        bank_a(to_integer(unsigned(addr_a))) <= chi_val;
      elsif (state = S_CLEAR) then
        bank_a(to_integer(unsigned(addr_a))) <= (others => '0');
      elsif (start = '1') and (state = S_IDLE) and (funct3(2 downto 1) = "00") and (lane_ok = '1') then
        bank_a(to_integer(unsigned(addr_a))) <= rs1_i;
      end if;
    end if;
  end process bank_a_proc;

  -- chi result (+ iota folded into lane 0) --
  chi_proc: process(r0, r1, r2, col, row, hbit, rnd)
    variable t : std_ulogic_vector(31 downto 0);
  begin
    t := r0 xor ((not r1) and r2);
    if (col = 0) and (row = 0) then -- iota: A[0] ^= round constant
      if (hbit = '0') then
        t := t xor rc_c(to_integer(rnd))(31 downto 0);
      else
        t := t xor rc_c(to_integer(rnd))(63 downto 32);
      end if;
    end if;
    chi_val <= t;
  end process chi_proc;

  bank_b_proc: process(clk_i)
  begin
    if rising_edge(clk_i) then
      if (state = S_RPWR) and (sub(0) = '0') then
        bank_b(to_integer(unsigned(addr_b))) <= rot64(31 downto 0);
      elsif (state = S_RPWR) and (sub(0) = '1') then
        bank_b(to_integer(unsigned(addr_b))) <= rot64(63 downto 32);
      end if;
    end if;
  end process bank_b_proc;

  c_ram_proc: process(clk_i)
  begin
    if rising_edge(clk_i) then
      if (state = S_THW) then
        if (hbit = '0') then
          c_ram(to_integer(unsigned(addr_c))) <= acl;
        else
          c_ram(to_integer(unsigned(addr_c))) <= ach;
        end if;
      end if;
    end if;
  end process c_ram_proc;

  -- asynchronous read ports --
  rd_a <= bank_a(to_integer(unsigned(addr_a)));
  rd_b <= bank_b(to_integer(unsigned(addr_b)));
  rd_c <= c_ram(to_integer(unsigned(addr_c)));

  -- Address Generation (combinational from counters / CPU decode) ---------------------------
  addr_proc: process(state, col, row, hbit, sub, funct3, lane_idx)
    variable l0, l1, l2 : integer;
  begin
    -- CPU port (idle): entry 2*lane + h ; funct3(0): 0 = LO half, 1 = HI half --
    addr_a <= std_ulogic_vector(to_unsigned(2*to_integer(unsigned(lane_idx)) + b2i(funct3(0)), 6));
    addr_b <= (others => '0');
    addr_c <= (others => '0');
    case state is
      when S_THA  => addr_a <= std_ulogic_vector(to_unsigned(2*(to_integer(col) + 5*to_integer(row)) + b2i(hbit), 6));
      when S_RPLD => addr_a <= std_ulogic_vector(to_unsigned(2*(to_integer(col) + 5*to_integer(row)) + b2i(sub(0)), 6));
      when S_CHWR => addr_a <= std_ulogic_vector(to_unsigned(2*(to_integer(col) + 5*to_integer(row)) + b2i(hbit), 6));
      when S_CLEAR=> addr_a <= std_ulogic_vector(to_unsigned(to_integer(sub), 6));
      when S_RPWR => addr_b <= std_ulogic_vector(to_unsigned(2*pi_dst(to_integer(col), to_integer(row)) + b2i(sub(0)), 6));
      when S_CHLD => -- read B(x,y), B(x+1,y), B(x+2,y)
        l0 := to_integer(col);
        l1 := (l0 + 1) mod 5;
        l2 := (l0 + 2) mod 5;
        case to_integer(sub) is
          when 0  => addr_b <= std_ulogic_vector(to_unsigned(2*(l0 + 5*to_integer(row)) + b2i(hbit), 6));
          when 1  => addr_b <= std_ulogic_vector(to_unsigned(2*(l1 + 5*to_integer(row)) + b2i(hbit), 6));
          when others => addr_b <= std_ulogic_vector(to_unsigned(2*(l2 + 5*to_integer(row)) + b2i(hbit), 6));
        end case;
      when S_THW  => addr_c <= std_ulogic_vector(to_unsigned(2*to_integer(col) + b2i(hbit), 4));
      when S_DLD  => -- C[x-1] (sub 0,1), C[x+1] (sub 2,3)
        if (sub(1) = '0') then
          addr_c <= std_ulogic_vector(to_unsigned(2*((to_integer(col)+4) mod 5) + b2i(sub(0)), 4));
        else
          addr_c <= std_ulogic_vector(to_unsigned(2*((to_integer(col)+1) mod 5) + b2i(sub(0)), 4));
        end if;
      when others => null;
    end case;
  end process addr_proc;

  -- Sequencer -------------------------------------------------------------------------------
  seq_proc: process(rstn_i, clk_i)
    variable q, s : integer;
  begin
    if (rstn_i = '0') then
      state <= S_IDLE;
      rnd <= (others => '0');
      col <= (others => '0');
      row <= (others => '0');
      sub <= (others => '0');
      hbit <= '0';
      done <= '0';
      acl <= (others => '0'); ach <= (others => '0');
      v64 <= (others => '0'); d64 <= (others => '0');
      cl64 <= (others => '0'); ch64 <= (others => '0');
      rot64 <= (others => '0');
      r0 <= (others => '0'); r1 <= (others => '0'); r2 <= (others => '0');
    elsif rising_edge(clk_i) then
      done <= '0'; -- default: single-cycle completion pulse
      case state is

        when S_IDLE =>
          if (start = '1') then
            case funct3 is
              when cfu_perm_c =>
                -- NOTE: rnd is NOT reset here. One PERM op advances the state by
                -- exactly one round; the round index persists across PERM ops and
                -- wraps automatically (0..23). A full Keccak-f1600 permutation is
                -- therefore 24 consecutive PERM ops (issued by the software wrapper).
                state <= S_THA;
                col <= (others => '0'); row <= (others => '0');
                sub <= (others => '0'); hbit <= '0';
              when cfu_clear_c =>
                -- a cleared state starts a fresh permutation sequence
                state <= S_CLEAR; sub <= (others => '0'); rnd <= (others => '0');
              when others => null;
            end case;
          end if;

        when S_CLEAR =>
          if (sub = 49) then
            state <= S_IDLE; done <= '1';
          else
            sub <= sub + 1;
          end if;

        -- theta: C[x] = XOR over y of A(x,y), accumulating a 32-bit entry per cycle --
        when S_THA =>
          if (hbit = '0') then
            if (row = 0) then acl <= rd_a; else acl <= acl xor rd_a; end if;
            hbit <= '1';
          else
            if (row = 0) then ach <= rd_a; else ach <= ach xor rd_a; end if;
            hbit <= '0';
            if (row = 4) then
              row <= (others => '0');
              state <= S_THW;
            else
              row <= row + 1;
            end if;
          end if;

        -- store the column parity into c_ram --
        when S_THW =>
          if (hbit = '0') then
            hbit <= '1';
          else
            hbit <= '0';
            if (col = 4) then
              col <= (others => '0');
              row <= (others => '0');
              state <= S_DLD;
            else
              col <= col + 1;
              state <= S_THA;
            end if;
          end if;

        -- load C[x-1] and C[x+1] --
        when S_DLD =>
          case to_integer(sub) is
            when 0 => cl64(31 downto 0)  <= rd_c;
            when 1 => cl64(63 downto 32) <= rd_c;
            when 2 => ch64(31 downto 0)  <= rd_c;
            when others => ch64(63 downto 32) <= rd_c;
          end case;
          if (sub = 3) then
            sub <= (others => '0');
            state <= S_DCM;
          else
            sub <= sub + 1;
          end if;

        -- D[x] = C[x-1] ^ rol1(C[x+1]) --
        when S_DCM =>
          d64 <= cl64 xor rol1(ch64);
          row <= (others => '0');
          state <= S_RPLD;

        -- read one whole lane (2 x 32 bit) from A --
        when S_RPLD =>
          if (sub(0) = '0') then
            v64(31 downto 0) <= rd_a;
            sub <= sub + 1;
          else
            v64(63 downto 32) <= rd_a;
            sub <= to_unsigned(3, 6); -- k = 3 (process chunks top-down, shift-up accumulator)
            rot64 <= (others => '0');
            state <= S_RPROT;
          end if;

        -- rho: rotate (v64 ^ d64) left by rho[col][row], one 16-bit chunk per cycle --
        when S_RPROT =>
          q := rho_q(to_integer(col), to_integer(row));
          s := rho_s(to_integer(col), to_integer(row));
          rot64 <= rot64(47 downto 0) &
                   funnel(chunk_of(v64 xor d64, to_integer(sub) - q),
                          chunk_of(v64 xor d64, to_integer(sub) - q - 1), s);
          if (sub = 0) then
            sub <= (others => '0');
            state <= S_RPWR;
          else
            sub <= sub - 1;
          end if;

        -- pi: write the rotated lane to B at the pi-permuted lane --
        when S_RPWR =>
          if (sub(0) = '0') then
            sub <= sub + 1;
          else
            sub <= (others => '0');
            if (row = 4) then
              row <= (others => '0');
              if (col = 4) then
                col <= (others => '0');
                state <= S_CHLD;
              else
                col <= col + 1;
                state <= S_DLD;
              end if;
            else
              row <= row + 1;
              state <= S_RPLD;
            end if;
          end if;

        -- chi: read B(x,y), B(x+1,y), B(x+2,y) --
        when S_CHLD =>
          case to_integer(sub) is
            when 0 => r0 <= rd_b;
            when 1 => r1 <= rd_b;
            when others => r2 <= rd_b;
          end case;
          if (sub = 2) then
            sub <= (others => '0');
            state <= S_CHWR;
          else
            sub <= sub + 1;
          end if;

        -- chi: write A(x,y) = B(x,y) ^ (~B(x+1,y) & B(x+2,y)); iota folded into lane 0 --
        when S_CHWR =>
          if (hbit = '0') then
            hbit <= '1';
            state <= S_CHLD;
          else
            hbit <= '0';
            if (col = 4) then
              col <= (others => '0');
              if (row = 4) then
                row <= (others => '0');
                state <= S_RND;
              else
                row <= row + 1;
                state <= S_CHLD;
              end if;
            else
              col <= col + 1;
              state <= S_CHLD;
            end if;
          end if;

        -- one round done: signal completion and advance/wrap the round counter --
        when S_RND =>
          state <= S_IDLE;
          done  <= '1';
          if (to_integer(rnd) = last_round_c) then
            rnd <= (others => '0'); -- completed round 23 -> next PERM starts a new permutation
          else
            rnd <= rnd + 1;
          end if;

        when others =>
          state <= S_IDLE;

      end case;
    end if;
  end process seq_proc;

  -- Function Result Select ------------------------------------------------------------------
  result_proc: process(op_valid, funct3, state, rd_a, done)
  begin
    result_o <= rd_a; -- read operations return the addressed bank entry
    valid_o  <= '0';
    if (op_valid = '1') then
      case funct3 is
        when cfu_read_lo_c | cfu_read_hi_c | cfu_write_lo_c | cfu_write_hi_c =>
          if (state = S_IDLE) then
            valid_o <= '1';
          else
            valid_o <= '0';
          end if;
        when cfu_perm_c | cfu_clear_c =>
          valid_o <= done;
        when others =>
          valid_o <= '0'; -- not handled by the Keccak engine (e.g. the NTT engine)
      end case;
    end if;
  end process result_proc;
end neorv32_cpu_alu_cfu_keccak_gen2;
