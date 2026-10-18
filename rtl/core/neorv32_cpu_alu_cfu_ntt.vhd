-- ================================================================================ --
-- NEORV32 CPU - ALU Custom Functions Unit (CFU) - NTT engine                        --
-- -------------------------------------------------------------------------------- --
-- Bulk polynomial NTT engine, shared by ML-KEM (FIPS 203, q = 3329) and             --
-- ML-DSA (FIPS 204, q = 8380417).                                                    --
--                                                                                  --
-- WHY A SCRATCHPAD: the NTT is memory-bound (256 coefficients in, 256 out). The      --
-- CFU ALU port is register-only, so a butterfly-per-instruction design would be       --
-- dominated by coefficient transfer, not arithmetic. Instead the CFU owns a 256 x    --
-- 32-bit coefficient scratchpad plus the twiddle-factor ROM; software streams the    --
-- polynomial in once, issues one op per NTT layer, and streams it out. The datapath  --
-- then iterates internally and only the (unavoidable) operand transfers cross the    --
-- register port (ML-KEM packs two int16 per word -> 2 x 128 ops; ML-DSA moves int32  --
-- -> 2 x 256 ops). This mirrors the "bulk/streaming over register                    --
-- shuffling" design rule learned from the Keccak CFU.                                --
--                                                                                  --
-- WHY THREE BANKS: the scratchpad MUST infer as distributed RAM (a flip-flop array   --
-- with four write sites and three asynchronous read ports was synthesizing to ~50k   --
-- LUTs of decoders and 256:1 read muxes -- more than the whole FPGA). A butterfly    --
-- needs TWO reads and TWO writes in the SAME cycle, but plain LUTRAM offers only     --
-- one read + one write port. Instead coefficient i is stored in bank (i mod 3) at    --
-- offset (i / 3): a butterfly always pairs indices (i, i+len) with len a power of    --
-- two (1..128), and 3 never divides a power of two, so the two operands ALWAYS land  --
-- in DIFFERENT banks. Three independent 86 x 32 single-port LUTRAMs therefore give    --
-- the butterfly its two reads and two writes at ~130 LUTs total.                      --
--                                                                                  --
-- REGISTERED ADDRESS DECODE: a naive (index / 3, index mod 3) decode placed in front  --
-- of the asynchronous LUTRAM read put a 9x9 `blk*2*len` multiply and two divide-by-3 --
-- networks on the read -> multiply path, which did not close timing at 50 MHz. The   --
-- (bank, offset) pair is therefore kept as a small REGISTER that is advanced          --
-- incrementally (x+1 and x+len are just a 2-bit modulo-3 add plus a 7-bit add), so    --
-- the path to the multiplier now starts at a register.                                --
--                                                                                  --
-- PIPELINE (3 cycles per butterfly + 1 flush; 128 butterflies per layer):            --
--   The three multiplies of the Montgomery reduction share ONE 32x32 multiplier, so  --
--   they must be issued in three consecutive cycles. The multiplier must never be    --
--   driven by the asynchronous LUTRAM read, NOR may its combinational output reach    --
--   the scratchpad write: either would make the synthesizer build a false path        --
--   spanning read -> mul1 -> mul_p -> mul3 -> reduce -> write, roughly twice the real --
--   depth (this missed timing by 4 ns; see the design log). The operands are          --
--   therefore latched into registers, the reduction result is registered, and each    --
--   butterfly's results are written one cycle later -- during the NEXT butterfly's    --
--   S_M1, which does not otherwise use the scratchpad write port:                     --
--     IDLE/LAYER : latch a,b of butterfly 0 (indices 0 and len)                       --
--     S_M1       : p1      <= a_b * zeta            (mul1)  + write back previous bf  --
--     S_M2       : m_red   <= low(p1) * QINV        (mul2)                            --
--     S_M3       : red64_r <= (p1 - m_red*q) >> s   (mul3 + reduce);                  --
--                  remember 'a' and the write addresses; latch a,b of the NEXT bf     --
--   One extra "flush" S_M1 after the last butterfly completes its deferred write.     --
--                                                                                  --
-- CORE PRIMITIVE: a dual-modulus Montgomery multiply  z = x * y * R^-1 mod q, with   --
--   ML-KEM: q = 3329,    R = 2^16, QINV = 62209                                       --
--   ML-DSA: q = 8380417, R = 2^32, QINV = 58728449                                     --
-- The low bits of a two's-complement product are identical for signed and unsigned    --
-- interpretation, so the same signed multiplier serves the unsigned QINV step.        --
--                                                                                  --
-- BUTTERFLY (DIT, identical structure in both libraries):                            --
--   t = montgomery_reduce(b * zeta) ; a' = a + t ; b' = a - t                         --
--                                                                                  --
-- LAYER SCHEDULE (matches mlk_ntt_layer / mld_ntt_layer exactly):                    --
--   for layer = 1..NL:  len = 256 >> layer ; k = 1 << (layer-1)                       --
--     for start = 0, 2*len, ... < 256:  zeta = zetas[k++]                            --
--       for j = start .. start+len-1:  butterfly(a = j, b = j+len)                   --
--   ML-KEM NL = 7, zetas[128]; ML-DSA NL = 8, zetas[256]. Every layer has exactly     --
--   128 butterflies, so a layer is 128*3 + 1 = 385 cycles of execution (386 measured  --
--   including the issue cycle) -- well inside the NEORV32 512-cycle co-processor       --
--   timeout.                                                                          --
--                                                                                  --
-- CFU instruction set (CUSTOM-0, R-type; funct3 = 110 selects this engine):          --
--   sub-op (funct7) = 0000000 -> CLEAR  : zero the scratchpad, reset ptr/layer        --
--   sub-op (funct7) = 0000001 -> MODE   : rs1[0] = 0 -> ML-KEM, 1 -> ML-DSA           --
--   sub-op (funct7) = 0000010 -> WRITE  : scratchpad[ptr] := rs1, ptr := ptr + step     --
--   sub-op (funct7) = 0000011 -> READ   : rd := scratchpad[ptr], ptr := ptr + step      --
--     step = coefficient granularity: ML-KEM packs TWO int16 into one 32-bit word       --
--     (step = 2; low half = coeff[2i], high half = coeff[2i+1]), so a transform moves   --
--     128 words; ML-DSA moves one int32 per op (step = 1, 256 words).                   --
--   sub-op (funct7) = 0000100 -> SETPTR : ptr := rs1                                  --
--   sub-op (funct7) = 0000101 -> LAYER  : run ONE forward NTT layer (multi-cycle)     --
--   The layer index lives in the CFU and advances/wraps (1..NL), so back-to-back      --
--   LAYER ops just work; MODE or CLEAR restarts at layer 1.                           --
--                                                                                  --
-- Round constants in neorv32_cpu_alu_cfu_ntt_zetas.vhd are AUTO-GENERATED from the   --
-- libraries' own zetas.inc so hardware and software use identical twiddles.           --
--                                                                                  --
-- The NEORV32 RISC-V Processor - https://github.com/stnolting/neorv32                --
-- Licensed under the BSD-3-Clause license, see LICENSE for details.                  --
-- SPDX-License-Identifier: BSD-3-Clause                                              --
-- ================================================================================ --

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library neorv32;
use neorv32.neorv32_package.all;
use neorv32.neorv32_cpu_alu_cfu_ntt_zetas.all;

entity neorv32_cpu_alu_cfu_ntt is
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
end neorv32_cpu_alu_cfu_ntt;

architecture neorv32_cpu_alu_cfu_ntt_rtl of neorv32_cpu_alu_cfu_ntt is

  -- instruction decode --
  constant opcode_custom0_c : std_ulogic_vector(6 downto 0) := "0001011"; -- CUSTOM-0 opcode
  constant funct3_ntt_c     : std_ulogic_vector(2 downto 0) := "110";     -- NTT engine select

  -- sub-opcodes (funct7) --
  constant sub_clear_c  : std_ulogic_vector(6 downto 0) := "0000000";
  constant sub_mode_c   : std_ulogic_vector(6 downto 0) := "0000001";
  constant sub_write_c  : std_ulogic_vector(6 downto 0) := "0000010";
  constant sub_read_c   : std_ulogic_vector(6 downto 0) := "0000011";
  constant sub_setptr_c : std_ulogic_vector(6 downto 0) := "0000100";
  constant sub_layer_c  : std_ulogic_vector(6 downto 0) := "0000101";

  -- ML-KEM modulus parameters (q = 3329, R = 2^16) --
  constant kem_q_c     : integer := 3329;
  constant kem_qinv_c  : integer := 62209;
  constant kem_shift_c : integer := 16;
  constant kem_nlay_c  : integer := 7;

  -- ML-DSA modulus parameters (q = 8380417, R = 2^32) --
  constant dsa_q_c     : integer := 8380417;
  constant dsa_qinv_c  : integer := 58728449;
  constant dsa_shift_c : integer := 32;
  constant dsa_nlay_c  : integer := 8;

  -- coefficient scratchpad: THREE interleaved banks, each 86 x 32-bit LUTRAM --
  -- coefficient index i lives in bank (i mod 3) at offset (i / 3) --
  constant bank_depth_c : integer := 86; -- ceil(256 / 3)
  type bank_t is array (0 to bank_depth_c-1) of std_ulogic_vector(31 downto 0);
  signal bank0, bank1, bank2 : bank_t;
  attribute ram_style : string;
  attribute ram_style of bank0, bank1, bank2 : signal is "distributed";

  -- (bank, offset) address pair: coefficient index = bank + 3*offset --
  -- bits (8 downto 7) = bank (0..2), bits (6 downto 0) = offset (0..85) --
  subtype bo_t is unsigned(8 downto 0);
  constant bo_zero_c : bo_t := (others => '0'); -- index 0
  constant bo_one_c  : bo_t := "010000000";     -- bank 1, offset 0 -> index 1
  constant bo_two_c  : bo_t := "100000000";     -- bank 2, offset 0 -> index 2

  -- add two (bank, offset) pairs: bank sums modulo 3, overflow carries into offset --
  function bo_add(a : bo_t; b : bo_t) return bo_t is
    variable s  : unsigned(2 downto 0);
    variable t  : unsigned(2 downto 0);
    variable ro : unsigned(6 downto 0);
    variable r  : bo_t;
  begin
    s  := resize(a(8 downto 7), 3) + resize(b(8 downto 7), 3); -- 0..4
    ro := a(6 downto 0) + b(6 downto 0);
    if (s >= 3) then
      t             := s - 3; -- 3->0, 4->1
      ro            := ro + 1;
      r(8 downto 7) := t(1 downto 0);
    else
      r(8 downto 7) := s(1 downto 0);
    end if;
    r(6 downto 0) := ro;
    return r;
  end function bo_add;

  -- decode a linear coefficient index into (bank, offset) -- off the critical path --
  function bo_of(idx : unsigned(8 downto 0)) return bo_t is
    variable r : bo_t;
  begin
    r(8 downto 7) := to_unsigned(to_integer(idx) mod 3, 2);
    r(6 downto 0) := to_unsigned(to_integer(idx) / 3, 7);
    return r;
  end function bo_of;

  -- instruction decode (combinational) --
  signal op_valid : std_ulogic;
  signal start    : std_ulogic;
  signal funct3   : std_ulogic_vector(2 downto 0);
  signal subop    : std_ulogic_vector(6 downto 0);

  -- sequencer --
  type state_t is (S_IDLE, S_CLEAR, S_M1, S_M2, S_M3);
  signal state : state_t;
  signal mode  : std_ulogic;               -- '0' = ML-KEM, '1' = ML-DSA
  signal ptr   : unsigned(8 downto 0);     -- coefficient stream pointer (0..256)
  signal layer : unsigned(3 downto 0);     -- current NTT layer (1..NL)
  signal blk   : unsigned(8 downto 0);     -- butterfly block within the current layer
  signal jj    : unsigned(8 downto 0);     -- butterfly within the current block
  signal len   : unsigned(8 downto 0);     -- block length = 256 >> layer
  signal nblk  : unsigned(8 downto 0);     -- #blocks = 1 << (layer-1) (= twiddle base)
  signal clr   : unsigned(8 downto 0);     -- CLEAR sweep counter
  signal done  : std_ulogic;

  -- control (combinational) --
  signal nlay      : unsigned(3 downto 0);
  signal len_nxt   : unsigned(8 downto 0); -- block length of the layer being started
  signal len_nxt_b : bo_t;
  signal is_last   : std_ulogic;           -- last butterfly of the layer

  -- twiddle factor --
  signal zeta_idx : unsigned(8 downto 0); -- ROM index = nblk + blk
  signal zeta_val : signed(31 downto 0);

  -- registered scratchpad address decode (bank, offset) --
  signal cur_a  : bo_t; -- current butterfly operand a (index j)
  signal cur_b  : bo_t; -- current butterfly operand b (index j + len)
  signal nxt_a  : bo_t; -- next butterfly operand a
  signal nxt_b  : bo_t; -- next butterfly operand b
  signal rd_a   : bo_t; -- selected read address (operand a slot)
  signal rd_b   : bo_t; -- selected read address (operand b slot)
  signal len_bo : bo_t; -- len as (bank, offset), latched per layer
  signal ptr_bo : bo_t; -- stream pointer decode
  signal clr_bo : bo_t; -- CLEAR sweep decode

  -- scratchpad ports --
  signal sel_a         : unsigned(1 downto 0); -- bank selected for the a/stream read
  signal ra0, ra1, ra2 : unsigned(6 downto 0); -- per-bank read address
  signal q0, q1, q2    : std_ulogic_vector(31 downto 0); -- per-bank read data
  signal d_a, d_b      : std_ulogic_vector(31 downto 0); -- operand a / b read data
  signal wr0_bo, wr1_bo : bo_t;                -- write slots (a' / b')
  signal w0a, w1a, w2a : unsigned(6 downto 0); -- per-bank write address
  signal w0d, w1d, w2d : std_ulogic_vector(31 downto 0); -- per-bank write data
  signal w0e, w1e, w2e : std_ulogic;           -- per-bank write enable
  signal wr0_en, wr1_en : std_ulogic;
  signal wr0_data, wr1_data : std_ulogic_vector(31 downto 0);

  -- shared multiplier (LUT-only by design: no DSP inference) --
  signal mul_a, mul_b : signed(31 downto 0);
  -- The multiplier is only 32 x 27: every B operand of every multiply fits in a
  -- 27-bit SIGNED value (the widest is ML-DSA mul2 with B = QINV = 58728449 < 2^26),
  -- so truncating mul_b to 27 bits is exact. Single-pass, so no B-halving FSM and no
  -- cycle cost: both modes keep the 3-cycle butterfly. Measured in the full design,
  -- this narrows the engine 2365 -> 2286 LUT and the critical path 2.81 -> 3.77 ns.
  signal mul_p        : signed(58 downto 0); -- combinational product (32 x 27)
  attribute use_dsp : string;
  attribute use_dsp of mul_p : signal is "no";

  -- butterfly datapath registers (multiplier operands are ALWAYS registered) --
  signal coeff_a : std_ulogic_vector(31 downto 0); -- 'a' operand of the current butterfly
  signal coeff_b : std_ulogic_vector(31 downto 0); -- 'b' operand of the current butterfly
  signal p1      : signed(63 downto 0);            -- mul1: x * y (64-bit signed product)
  signal m_red   : unsigned(63 downto 0);          -- mul2: low(x*y) * QINV

  -- deferred-write registers (break the multiplier -> write false path) --
  signal red64_c : signed(63 downto 0);            -- (x*y - t*q) >> shift (combinational)
  signal red64_r : signed(63 downto 0);            -- registered reduction result
  signal qa_r    : std_ulogic_vector(31 downto 0); -- this butterfly's 'a', kept for +/- t
  signal wra_r   : bo_t;                           -- deferred write address (a')
  signal wrb_r   : bo_t;                           -- deferred write address (b')
  signal wr_pend : std_ulogic;                     -- a butterfly result is waiting to be written
  signal last_bf : std_ulogic;                     -- last butterfly: one flush cycle left

  -- reduction / write datapath (combinational, consumed during S_M1) --
  signal rm_r  : signed(31 downto 0); -- reduced Montgomery product
  signal sum_i : signed(31 downto 0); -- qa_r + rm_r
  signal dif_i : signed(31 downto 0); -- qa_r - rm_r
  signal sum_v : std_ulogic_vector(31 downto 0);
  signal dif_v : std_ulogic_vector(31 downto 0);

begin

  -- Instruction Decode ---------------------------------------------------------------------
  op_valid <= '1' when (inst_i(6 downto 0) = opcode_custom0_c) and (inst_i(14 downto 12) = funct3_ntt_c) else '0';
  start    <= start_i when (op_valid = '1') else '0';
  funct3   <= inst_i(14 downto 12);
  subop    <= inst_i(31 downto 25);

  -- Number of layers depends on the modulus ------------------------------------------------
  nlay <= to_unsigned(kem_nlay_c, nlay'length) when (mode = '0') else to_unsigned(dsa_nlay_c, nlay'length);

  -- Layer Geometry (combinational, latched at layer start) ---------------------------------
  len_nxt   <= shift_right(to_unsigned(256, len_nxt'length), to_integer(layer));
  len_nxt_b <= bo_of(len_nxt);
  is_last   <= '1' when (jj = len - 1) and (blk = nblk - 1) else '0';

  -- Twiddle Factor Select -------------------------------------------------------------------
  -- zeta index = (1 << (layer-1)) + blk == nblk + blk. The ROM is a table of integers (see
  -- neorv32_cpu_alu_cfu_ntt_zetas.vhd); convert to the signed datapath type here.
  zeta_idx <= nblk + blk;
  zeta_proc: process(mode, zeta_idx)
  begin
    if (mode = '0') then -- ML-KEM: 128 int16 twiddles (index 0..127)
      zeta_val <= to_signed(zetas_kem_c(to_integer(zeta_idx(6 downto 0))), 32);
    else                 -- ML-DSA: 256 int32 twiddles (index 0..255)
      zeta_val <= to_signed(zetas_dsa_c(to_integer(zeta_idx(7 downto 0))), 32);
    end if;
  end process zeta_proc;

  -- Next Butterfly Address (combinational; registered at the end of S_M3) -------------------
  -- Within a block both operands advance by 1; at a block boundary they jump by len+1. Because
  -- cur_b = cur_a + len, "jump to the next block" is simply (cur_b + 1).
  nxt_a <= bo_add(cur_b, bo_one_c) when (jj = len - 1) else bo_add(cur_a, bo_one_c);
  nxt_b <= bo_add(nxt_a, len_bo);

  -- Read Address Select (combinational from registers only) ---------------------------------
  -- S_M3 reads the NEXT butterfly's operands (this butterfly's read/write ports are separate,
  -- so the overlap is free); the last butterfly of a layer must not issue the out-of-range
  -- next address, so it re-reads its own (harmless -- the result is only latched, not used).
  rd_sel_proc: process(state, start, subop, is_last, cur_a, cur_b, nxt_a, nxt_b, ptr_bo, len_nxt_b)
  begin
    if (state = S_M3) then
      if (is_last = '0') then
        rd_a <= nxt_a;
        rd_b <= nxt_b;
      else
        rd_a <= cur_a;
        rd_b <= cur_b;
      end if;
    elsif (state = S_M1) or (state = S_M2) then
      rd_a <= cur_a;
      rd_b <= cur_b;
    elsif (state = S_IDLE) and (start = '1') and (subop = sub_layer_c) then
      rd_a <= bo_zero_c;    -- first butterfly: a = index 0
      rd_b <= len_nxt_b;    --                    b = index len
    else
      rd_a <= ptr_bo;                     -- stream read (READ op): word at ptr
      rd_b <= bo_add(ptr_bo, bo_one_c);   -- partner word ptr+1 (ML-KEM pairing)
    end if;
  end process rd_sel_proc;

  -- Read Address Steer (bank = index mod 3) -------------------------------------------------
  rd_steer_proc: process(rd_a, rd_b)
  begin
    ra0 <= (others => '0');
    ra1 <= (others => '0');
    ra2 <= (others => '0');
    if (rd_a(8 downto 7) = 0) then ra0 <= rd_a(6 downto 0); elsif (rd_a(8 downto 7) = 1) then ra1 <= rd_a(6 downto 0); else ra2 <= rd_a(6 downto 0); end if;
    if (rd_b(8 downto 7) = 0) then ra0 <= rd_b(6 downto 0); elsif (rd_b(8 downto 7) = 1) then ra1 <= rd_b(6 downto 0); else ra2 <= rd_b(6 downto 0); end if;
  end process rd_steer_proc;

  -- Per-bank asynchronous read + operand select ---------------------------------------------
  q0 <= bank0(to_integer(ra0));
  q1 <= bank1(to_integer(ra1));
  q2 <= bank2(to_integer(ra2));

  sel_a <= rd_a(8 downto 7);
  d_a   <= q0 when (sel_a = 0) else q1 when (sel_a = 1) else q2; -- operand a / stream read data
  d_b   <= q0 when (rd_b(8 downto 7) = 0) else q1 when (rd_b(8 downto 7) = 1) else q2;

  -- Write Data / Enable ---------------------------------------------------------------------
  -- The butterfly result is written one cycle after it is computed (S_M1 of the next
  -- butterfly), when the scratchpad write port is otherwise idle.
  -- In ML-KEM mode a WRITE carries TWO coefficients: slot 0 gets coeff[2i] (the low half
  -- of rs1, sign-extended) at index ptr, slot 1 gets coeff[2i+1] (the high half) at
  -- index ptr+1. Those two indices are in DIFFERENT banks, so this rides for free on
  -- the butterfly's two write slots.
  wr0_bo <= wra_r when (state = S_M1) else ptr_bo when (state = S_IDLE) else clr_bo;
  wr1_bo <= wrb_r when (state = S_M1) else bo_add(ptr_bo, bo_one_c);
  wr0_en <= '1' when ((state = S_M1) and (wr_pend = '1')) or (state = S_CLEAR) or
                      ((state = S_IDLE) and (start = '1') and (subop = sub_write_c)) else '0';
  wr1_en <= '1' when ((state = S_M1) and (wr_pend = '1')) or
                     ((state = S_IDLE) and (start = '1') and (subop = sub_write_c) and (mode = '0')) else '0';
  wr0_data <= sum_v when (state = S_M1) else
              std_ulogic_vector(resize(signed(rs1_i(15 downto 0)), 32)) when ((subop = sub_write_c) and (mode = '0')) else
              rs1_i when (subop = sub_write_c) else
              (others => '0'); -- CLEAR writes zeros
  wr1_data <= dif_v when (state = S_M1) else
              std_ulogic_vector(resize(signed(rs1_i(31 downto 16)), 32)) when ((subop = sub_write_c) and (mode = '0')) else
              (others => '0');

  -- Write Address Steer (combinational from registers only) ---------------------------------
  -- A butterfly's a' and b' always target DIFFERENT banks (a and b never collide), so at most
  -- one write slot per bank.
  wr_addr_proc: process(wr0_en, wr1_en, wr0_bo, wr1_bo, wr0_data, wr1_data)
  begin
    w0e <= '0'; w1e <= '0'; w2e <= '0';
    w0a <= (others => '0'); w1a <= (others => '0'); w2a <= (others => '0');
    w0d <= (others => '0'); w1d <= (others => '0'); w2d <= (others => '0');
    if (wr0_en = '1') then
      if (wr0_bo(8 downto 7) = 0) then w0e <= '1'; w0a <= wr0_bo(6 downto 0); w0d <= wr0_data;
      elsif (wr0_bo(8 downto 7) = 1) then w1e <= '1'; w1a <= wr0_bo(6 downto 0); w1d <= wr0_data;
      else w2e <= '1'; w2a <= wr0_bo(6 downto 0); w2d <= wr0_data; end if;
    end if;
    if (wr1_en = '1') then
      if (wr1_bo(8 downto 7) = 0) then w0e <= '1'; w0a <= wr1_bo(6 downto 0); w0d <= wr1_data;
      elsif (wr1_bo(8 downto 7) = 1) then w1e <= '1'; w1a <= wr1_bo(6 downto 0); w1d <= wr1_data;
      else w2e <= '1'; w2a <= wr1_bo(6 downto 0); w2d <= wr1_data; end if;
    end if;
  end process wr_addr_proc;

  -- Coefficient Scratchpad (distributed-RAM inference; no reset on the arrays!) -------------
  bank_proc: process(clk_i)
  begin
    if rising_edge(clk_i) then
      if (w0e = '1') then bank0(to_integer(w0a)) <= w0d; end if;
      if (w1e = '1') then bank1(to_integer(w1a)) <= w1d; end if;
      if (w2e = '1') then bank2(to_integer(w2a)) <= w2d; end if;
    end if;
  end process bank_proc;

  -- Shared Multiplier Operand Select ---------------------------------------------------------
  -- Only REGISTERED values may reach the multiplier; a combinational LUTRAM read here would
  -- create a false read -> mul1 -> reduce -> write path that spans the whole pipeline.
  mul_proc: process(state, mode, coeff_b, p1, m_red, zeta_val)
  begin
    mul_a <= (others => '0');
    mul_b <= (others => '0');
    case state is
      when S_M1 => -- mul1: coefficient * twiddle
        mul_a <= signed(coeff_b);
        mul_b <= zeta_val;
      when S_M2 => -- mul2: low(x*y) * QINV
        if (mode = '0') then
          mul_a <= signed(resize(unsigned(p1(15 downto 0)), 32));
          mul_b <= to_signed(kem_qinv_c, 32);
        else
          mul_a <= p1(31 downto 0);
          mul_b <= to_signed(dsa_qinv_c, 32);
        end if;
      when S_M3 => -- mul3: t * q
        if (mode = '0') then
          mul_a <= resize(signed(m_red(15 downto 0)), 32);
          mul_b <= to_signed(kem_q_c, 32);
        else
          mul_a <= signed(m_red(31 downto 0));
          mul_b <= to_signed(dsa_q_c, 32);
        end if;
      when others =>
        null;
    end case;
  end process mul_proc;
  mul_p <= mul_a * mul_b(26 downto 0); -- 32 x 27 -> 59-bit signed product

  -- Montgomery Reduction (combinational product, REGISTERED result) --------------------------
  -- red64 = (x*y - t*q) >> shift ; the register is what keeps the multiplier output off the
  -- scratchpad write-data path.
  red64_c <= shift_right(p1 - resize(mul_p, 64), kem_shift_c) when (mode = '0') else
             shift_right(p1 - resize(mul_p, 64), dsa_shift_c);

  rm_r <= resize(signed(red64_r(15 downto 0)), 32) when (mode = '0') else red64_r(31 downto 0);

  -- ML-KEM stores int16 coefficients, so the butterfly add/sub is truncated to 16 bits just
  -- like the C reference. ML-DSA is int32 throughout and wraps naturally at 32 bits.
  sum_i <= signed(qa_r) + rm_r;
  dif_i <= signed(qa_r) - rm_r;
  sum_v <= std_ulogic_vector(resize(sum_i(15 downto 0), 32)) when (mode = '0') else std_ulogic_vector(sum_i);
  dif_v <= std_ulogic_vector(resize(dif_i(15 downto 0), 32)) when (mode = '0') else std_ulogic_vector(dif_i);

  -- Sequencer -------------------------------------------------------------------------------
  seq_proc: process(rstn_i, clk_i)
  begin
    if (rstn_i = '0') then
      state   <= S_IDLE;
      mode    <= '0';
      ptr     <= (others => '0');
      layer   <= to_unsigned(1, layer'length);
      blk     <= (others => '0');
      jj      <= (others => '0');
      len     <= (others => '0');
      nblk    <= (others => '0');
      clr     <= (others => '0');
      cur_a   <= bo_zero_c;
      cur_b   <= bo_zero_c;
      len_bo  <= bo_zero_c;
      ptr_bo  <= bo_zero_c;
      clr_bo  <= bo_zero_c;
      coeff_a <= (others => '0');
      coeff_b <= (others => '0');
      p1      <= (others => '0');
      m_red   <= (others => '0');
      red64_r <= (others => '0');
      qa_r    <= (others => '0');
      wra_r   <= bo_zero_c;
      wrb_r   <= bo_zero_c;
      wr_pend <= '0';
      last_bf <= '0';
      done    <= '0';
    elsif rising_edge(clk_i) then
      done <= '0'; -- default: single-cycle completion pulse
      case state is

        when S_IDLE =>
          if (start = '1') then
            case subop is
              when sub_clear_c =>
                state  <= S_CLEAR;
                clr    <= (others => '0');
                clr_bo <= bo_zero_c;
                ptr    <= (others => '0');
                ptr_bo <= bo_zero_c;
                layer  <= to_unsigned(1, layer'length);
              when sub_mode_c =>
                mode  <= rs1_i(0);
                layer <= to_unsigned(1, layer'length); -- MODE restarts at layer 1
                done  <= '1';
              when sub_write_c =>
                if (mode = '0') then -- ML-KEM: one word = two packed int16
                  ptr    <= ptr + 2;
                  ptr_bo <= bo_add(ptr_bo, bo_two_c);
                else
                  ptr    <= ptr + 1;
                  ptr_bo <= bo_add(ptr_bo, bo_one_c);
                end if;
                done   <= '1';
              when sub_read_c =>
                if (mode = '0') then -- ML-KEM: one word = two packed int16
                  ptr    <= ptr + 2;
                  ptr_bo <= bo_add(ptr_bo, bo_two_c);
                else
                  ptr    <= ptr + 1;
                  ptr_bo <= bo_add(ptr_bo, bo_one_c);
                end if;
                done   <= '1';
              when sub_setptr_c =>
                ptr    <= unsigned(rs1_i(8 downto 0));
                ptr_bo <= bo_of(unsigned(rs1_i(8 downto 0)));
                done   <= '1';
              when sub_layer_c =>
                -- fetch the layer geometry and latch butterfly 0's operands
                len     <= len_nxt;
                nblk    <= shift_left(to_unsigned(1, nblk'length), to_integer(layer) - 1);
                len_bo  <= len_nxt_b;
                blk     <= (others => '0');
                jj      <= (others => '0');
                cur_a   <= bo_zero_c;             -- first butterfly: a = block base (index 0)
                cur_b   <= len_nxt_b;             --                    b = a + len
                coeff_a <= d_a;                   -- latch operand a (index 0)
                coeff_b <= d_b;                   -- latch operand b (index len)
                state   <= S_M1;
              when others =>
                null;
            end case;
          end if;

        when S_CLEAR =>
          if (clr = 255) then
            state <= S_IDLE;
            done  <= '1';
          else
            clr    <= clr + 1;
            clr_bo <= bo_add(clr_bo, bo_one_c);
          end if;

        -- mul1: x*y, plus the deferred write of the previous butterfly (if any) ------------
        when S_M1 =>
          p1      <= resize(mul_p, 64);
          wr_pend <= '0';
          if (last_bf = '1') then
            state   <= S_IDLE;
            done    <= '1';
            last_bf <= '0';
          else
            state <= S_M2;
          end if;

        -- mul2: low(x*y) * QINV ----------------------------------------------------------
        when S_M2 =>
          m_red <= unsigned(resize(mul_p, 64));
          state <= S_M3;

        -- mul3 + reduce (registered); remember write info; prefetch the NEXT butterfly -----
        when S_M3 =>
          red64_r <= red64_c;
          qa_r    <= coeff_a; -- this butterfly's 'a' (before the prefetch below overwrites it)
          wra_r   <= cur_a;
          wrb_r   <= cur_b;
          wr_pend <= '1';
          coeff_a <= d_a; -- prefetch the NEXT butterfly's operand a (the read of nxt_a/nxt_b is
          coeff_b <= d_b; -- issued this cycle and the read/write ports are separate)
          if (jj = len - 1) then
            jj <= (others => '0');
            if (blk = nblk - 1) then -- layer finished
              last_bf <= '1';
              state   <= S_M1;      -- one flush cycle to complete the deferred write
              if (layer = nlay) then
                layer <= to_unsigned(1, layer'length); -- wrap: back-to-back LAYER ops
              else
                layer <= layer + 1;
              end if;
            else
              blk   <= blk + 1;
              cur_a <= nxt_a;
              cur_b <= nxt_b;
              state <= S_M1;
            end if;
          else
            jj    <= jj + 1;
            cur_a <= nxt_a;
            cur_b <= nxt_b;
            state <= S_M1;
          end if;

        when others =>
          state <= S_IDLE;

      end case;
    end if;
  end process seq_proc;

  -- Function Result Select ------------------------------------------------------------------
  -- The low half of a READ always comes from the word at ptr. In ML-KEM mode that word
  -- holds two packed int16 coefficients, so the high half comes from the partner word
  -- (ptr + 1) reached through rd_b / d_b; in ML-DSA mode it is the top half of the same
  -- word. Everything else the CPU could capture is don't-care.
  result_o(15 downto 0)  <= d_a(15 downto 0);
  result_o(31 downto 16) <= d_b(15 downto 0) when ((state = S_IDLE) and (mode = '0') and (subop = sub_read_c)) else
                            d_a(31 downto 16);

  valid_proc: process(op_valid, state, subop, done)
  begin
    valid_o <= '0';
    if (op_valid = '1') then
      case subop is
        when sub_write_c | sub_read_c | sub_mode_c | sub_setptr_c =>
          if (state = S_IDLE) then
            valid_o <= '1';
          else
            valid_o <= '0';
          end if;
        when sub_clear_c | sub_layer_c =>
          valid_o <= done;
        when others =>
          valid_o <= '0'; -- unsupported sub-op
      end case;
    end if;
  end process valid_proc;

end neorv32_cpu_alu_cfu_ntt_rtl;
