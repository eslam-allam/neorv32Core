-- ================================================================================ --
-- NEORV32 CPU - ALU Custom Functions Unit (CFU) - Keccak-f1600 permutation engine   --
-- -------------------------------------------------------------------------------- --
-- See the CPU's data sheet (docs/datasheet/cpu_cfu.adoc) for the CFU interface.     --
-- -------------------------------------------------------------------------------- --
-- This CFU replaces the upstream XTEA example. It is a stateful Keccak-f1600        --
-- (FIPS 202) permutation engine, shared by ML-KEM (FIPS 203) and ML-DSA (FIPS 204). --
--                                                                                  --
-- Rationale: RV32 has no 64-bit registers / bitmanip, so the portable C Keccak      --
-- permutation is extremely expensive on this core. The CFU owns the full 1600-bit   --
-- state (25 lanes x 64 bit) and performs one complete 24-round permutation per      --
-- CUSTOM-0 instruction. Software loads the state, issues the permutation and reads  --
-- the state back -- see sw/projects/user_libs/cfu_keccak.h and the FIPS202 native   --
-- backend headers for the software counterpart.                                     --
--                                                                                  --
-- Lane indexing follows FIPS 202 / the mlkem-native layout: lane index = x + 5*y,   --
-- i.e. state words [0..24] map 1:1 onto the library's uint64_t state[25].           --
-- Within a lane, little-endian: word index 2*L = lane L bits [31:0], 2*L+1 = [63:32] --
-- (matching how a uint64_t is laid out in RV32 memory).                             --
--                                                                                  --
-- CFU instruction set (all CUSTOM-0, R-type):                                       --
--   lane index is taken from rs2 (register value, 0..24); rs1 = data in / rd = out  --
--   funct3 = 000 -> WRITE_LO : lane[rs2][31:0]  := rs1                              --
--   funct3 = 001 -> WRITE_HI : lane[rs2][63:32] := rs1                              --
--   funct3 = 010 -> READ_LO  : rd := lane[rs2][31:0]                                --
--   funct3 = 011 -> READ_HI  : rd := lane[rs2][63:32]                               --
--   funct3 = 100 -> PERM     : perform 24 Keccak-f1600 rounds (multi-cycle, ~25 cy) --
--   funct3 = 101 -> CLEAR    : zero the whole 1600-bit state                        --
--   all other funct3 values return valid_o = '0' -> illegal instruction trap        --
--                                                                                  --
-- R-type (rather than I-type) is used so the lane index can be a *register*         --
-- operand -- a software loop counter cannot be an I-type immediate under the        --
-- ".insn" intrinsic's "i" constraint.                                               --
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

library neorv32;
use neorv32.neorv32_package.all;

entity neorv32_cpu_alu_cfu is
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
end neorv32_cpu_alu_cfu;

architecture neorv32_cpu_alu_cfu_rtl of neorv32_cpu_alu_cfu is

  -- instruction decode (opcode field, reconstructed by the ALU) --
  constant opcode_custom0_c : std_ulogic_vector(6 downto 0) := "0001011"; -- CUSTOM-0 opcode

  -- instruction identifiers (funct3 bit-field) --
  constant cfu_write_lo_c : std_ulogic_vector(2 downto 0) := "000";
  constant cfu_write_hi_c : std_ulogic_vector(2 downto 0) := "001";
  constant cfu_read_lo_c  : std_ulogic_vector(2 downto 0) := "010";
  constant cfu_read_hi_c  : std_ulogic_vector(2 downto 0) := "011";
  constant cfu_perm_c     : std_ulogic_vector(2 downto 0) := "100";
  constant cfu_clear_c    : std_ulogic_vector(2 downto 0) := "101";

  -- Keccak state: 25 lanes x 64 bit; lane index = x + 5*y (FIPS 202) --
  type lane_array_t is array (0 to 24) of std_ulogic_vector(63 downto 0);
  type row_array_t  is array (0 to 4)  of std_ulogic_vector(63 downto 0);

  -- rho rotation offsets r[x][y] --
  type rho_array_t is array (0 to 4, 0 to 4) of natural;
  constant rho_offsets_c : rho_array_t := (
    ( 0, 36,  3, 41, 18),
    ( 1, 44, 10, 45,  2),
    (62,  6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39,  8, 14)
  );

  -- iota round constants --
  type rc_array_t is array (0 to 23) of std_ulogic_vector(63 downto 0);
  constant round_constants_c : rc_array_t := (
    x"0000000000000001", x"0000000000008082", x"800000000000808a", x"8000000080008000",
    x"000000000000808b", x"0000000080000001", x"8000000080008081", x"8000000000008009",
    x"000000000000008a", x"0000000000000088", x"0000000080008009", x"000000008000000a",
    x"000000008000808b", x"800000000000008b", x"8000000000008089", x"8000000000008003",
    x"8000000000008002", x"8000000000000080", x"000000000000800a", x"800000008000000a",
    x"8000000080008081", x"8000000000008080", x"0000000080000001", x"8000000080008008"
  );

  -- 64-bit rotate-left by a (elaboration-time constant) amount --
  function rol64(v : std_ulogic_vector(63 downto 0); n : natural) return std_ulogic_vector is
    variable u : unsigned(63 downto 0);
  begin
    if (n = 0) then
      return v;
    else
      u := unsigned(v);
      return std_ulogic_vector(shift_left(u, n) or shift_right(u, 64 - n));
    end if;
  end function rol64;

  -- one Keccak-f1600 round (theta, rho, pi, chi, iota) --
  function keccak_round(a : lane_array_t; rc : std_ulogic_vector(63 downto 0)) return lane_array_t is
    variable c, d : row_array_t;
    variable b    : lane_array_t;
    variable res  : lane_array_t;
    variable i, j : integer;
  begin
    -- theta: C[x] = XOR of column x, then D[x] = C[x-1] ^ rol(C[x+1], 1) --
    for x in 0 to 4 loop
      c(x) := a(x) xor a(x+5) xor a(x+10) xor a(x+15) xor a(x+20);
    end loop;
    for x in 0 to 4 loop
      d(x) := c((x+4) mod 5) xor rol64(c((x+1) mod 5), 1);
    end loop;
    -- rho + pi: B[y][(2x+3y) mod 5] = rol(A[x][y] ^ D[x], r[x][y]) --
    for x in 0 to 4 loop
      for y in 0 to 4 loop
        i := y;
        j := (2*x + 3*y) mod 5;
        b(i + 5*j) := rol64(a(x + 5*y) xor d(x), rho_offsets_c(x, y));
      end loop;
    end loop;
    -- chi: A[x][y] = B[x][y] ^ ((not B[x+1][y]) and B[x+2][y]) --
    for x in 0 to 4 loop
      for y in 0 to 4 loop
        res(x + 5*y) := b(x + 5*y) xor ((not b(((x+1) mod 5) + 5*y)) and b(((x+2) mod 5) + 5*y));
      end loop;
    end loop;
    -- iota --
    res(0) := res(0) xor rc;
    return res;
  end function keccak_round;

  -- instruction decoder --
  signal op_valid : std_ulogic; -- valid CUSTOM-0 (R-type) CFU instruction
  signal start    : std_ulogic; -- gated start trigger
  signal funct3   : std_ulogic_vector(2 downto 0);
  signal lane_idx : std_ulogic_vector(4 downto 0); -- rs2(4:0)
  signal lane_ok  : std_ulogic;                    -- lane index in range (0..24)

  -- Keccak engine --
  signal state_reg : lane_array_t;
  signal perm_busy : std_ulogic;                 -- permutation in progress
  signal perm_done : std_ulogic;                 -- one-cycle completion pulse
  signal round_cnt : unsigned(4 downto 0);       -- 0..23

begin

  -- Instruction Decode ---------------------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  op_valid <= '1' when (inst_i(6 downto 0) = opcode_custom0_c) else '0';
  start    <= start_i when (op_valid = '1') else '0';
  funct3   <= inst_i(14 downto 12); -- function select
  lane_idx <= rs2_i(4 downto 0);    -- lane index (register rs2)
  lane_ok  <= '1' when (unsigned(rs2_i) < 25) else '0';


  -- Keccak Processing Core ------------------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  keccak_core: process(rstn_i, clk_i)
  begin
    if (rstn_i = '0') then
      state_reg <= (others => (others => '0'));
      perm_busy <= '0';
      perm_done <= '0';
      round_cnt <= (others => '0');
    elsif rising_edge(clk_i) then
      perm_done <= '0'; -- default: no completion pulse
      if (start = '1') then
        case funct3 is
          when cfu_write_lo_c => -- write low 32-bit half of a lane
            if (lane_ok = '1') then
              state_reg(to_integer(unsigned(lane_idx)))(31 downto 0) <= rs1_i;
            end if;
          when cfu_write_hi_c => -- write high 32-bit half of a lane
            if (lane_ok = '1') then
              state_reg(to_integer(unsigned(lane_idx)))(63 downto 32) <= rs1_i;
            end if;
          when cfu_clear_c => -- zero the whole state
            state_reg <= (others => (others => '0'));
          when cfu_perm_c => -- start the 24-round permutation
            perm_busy <= '1';
            round_cnt <= (others => '0');
          when others => -- read operations do not modify the state
            null;
        end case;
      elsif (perm_busy = '1') then
        state_reg <= keccak_round(state_reg, round_constants_c(to_integer(round_cnt)));
        if (round_cnt = 23) then -- last round done
          perm_busy <= '0';
          perm_done <= '1';
          round_cnt <= (others => '0');
        else
          round_cnt <= round_cnt + 1;
        end if;
      end if;
    end if;
  end process keccak_core;


  -- Function Result Select ------------------------------------------------------------------
  -- -------------------------------------------------------------------------------------------
  result_select: process(op_valid, funct3, lane_idx, lane_ok, state_reg, perm_done)
  begin -- no register stage here; the CFU result is registered inside the ALU module
    result_o <= (others => '0');
    valid_o  <= '0';
    if (op_valid = '1') then
      case funct3 is
        when cfu_read_lo_c => -- read low half
          if (lane_ok = '1') then
            result_o <= state_reg(to_integer(unsigned(lane_idx)))(31 downto 0);
          end if;
          valid_o <= '1';
        when cfu_read_hi_c => -- read high half
          if (lane_ok = '1') then
            result_o <= state_reg(to_integer(unsigned(lane_idx)))(63 downto 32);
          end if;
          valid_o <= '1';
        when cfu_write_lo_c | cfu_write_hi_c | cfu_clear_c => -- write/clear complete immediately
          valid_o <= '1';
        when cfu_perm_c => -- multi-cycle permutation
          valid_o <= perm_done;
        when others => -- unspecified operation -> illegal instruction exception
          valid_o <= '0';
      end case;
    end if;
  end process result_select;

end neorv32_cpu_alu_cfu_rtl;
