-- ================================================================================ --
-- NEORV32 PQC - CFU dispatcher (Keccak-f1600 + forward/inverse NTT engines)
-- -------------------------------------------------------------------------------- --
-- One CFU co-processor slot, several engines selected by funct3:
--
--   funct3 = 000..101 : Keccak-f1600 engine (see neorv32_cpu_alu_cfu_keccak*.vhd)
--                       000/001 = write lo/hi, 010/011 = read lo/hi, 100 = PERM, 101 = CLEAR
--   funct3 = 110      : NTT engine        (see neorv32_cpu_alu_cfu_ntt.vhd)
--   funct3 = 111      : rejection sampler (see neorv32_cpu_alu_cfu_rej.vhd)
--
-- NOTE: 010 and 011 are already Keccak READ_LO / READ_HI, so the NTT and
--       rejection engines must use 110/111 - do NOT move them into 000..101.
--
-- The Keccak engine exists in two generations and the generic CFU_KECCAK_GEN
-- selects which one is elaborated:
--
--   1 - wide datapath, one round per clock, 1600-bit state in flip-flops.
--        A single PERM op performs the whole 24-round permutation (~26 cycles).
--        Area: 3753 LUT / 1607 FF.
--   2 - area-optimized: 16-bit serialized datapath, state in distributed RAM.
--        A PERM op advances exactly ONE round (~487 cycles), so a permutation is
--        24 ops. Area: 678 LUT / 503 FF.
--
-- The NTT engine is optional and controlled by CPU_CFU_NTT_EN.
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

entity neorv32_cpu_alu_cfu is
  generic (
    -- Keccak engine generation: 1 = wide (1 PERM op = whole permutation),
    -- 2 = serialized (1 PERM op = 1 round). Must match the software wrapper.
    CFU_KECCAK_GEN : natural := 2;
    -- NTT engine enable: when false, the NTT engine is not instantiated and
    -- funct3=110 is treated as illegal instruction.
    CPU_CFU_NTT_EN : boolean := true;
    -- NTT multiplier implementation: true = use DSP48E1 (Xilinx 7-series),
    -- false = behavioral multiplier (tool decides: LUTs or inferred DSP)
    CPU_CFU_NTT_USE_DSP : boolean := false
  );
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

  -- engine select --
  constant cfu_keccak_c : std_ulogic_vector(2 downto 0) := "000"; -- Keccak base (000..101)
  constant cfu_ntt_c    : std_ulogic_vector(2 downto 0) := "110"; -- NTT engine
  constant cfu_rej_c    : std_ulogic_vector(2 downto 0) := "111"; -- rejection-sampling engine

  -- accelerator engines: one CFU slot, several engines selected by funct3 --
  signal keccak_result_o : std_ulogic_vector(31 downto 0);
  signal keccak_valid_o  : std_ulogic;
  signal ntt_result_o    : std_ulogic_vector(31 downto 0);
  signal ntt_valid_o     : std_ulogic;
  signal rej_result_o    : std_ulogic_vector(31 downto 0);
  signal rej_valid_o     : std_ulogic;
  signal funct3          : std_ulogic_vector(2 downto 0);

begin

  funct3 <= inst_i(14 downto 12);

  -- Keccak-f1600 engine (generation selected by CFU_KECCAK_GEN) ----------------------
  u_keccak_gen1: if CFU_KECCAK_GEN = 1 generate
    keccak: entity neorv32.neorv32_cpu_alu_cfu_keccak(neorv32_cpu_alu_cfu_keccak_gen1)
      port map (
        clk_i    => clk_i,
        rstn_i   => rstn_i,
        start_i  => start_i,
        inst_i   => inst_i,
        rs1_i    => rs1_i,
        rs2_i    => rs2_i,
        result_o => keccak_result_o,
        valid_o  => keccak_valid_o
      );
  end generate;

  u_keccak_gen2: if CFU_KECCAK_GEN = 2 generate
    keccak: entity neorv32.neorv32_cpu_alu_cfu_keccak(neorv32_cpu_alu_cfu_keccak_gen2)
      port map (
        clk_i    => clk_i,
        rstn_i   => rstn_i,
        start_i  => start_i,
        inst_i   => inst_i,
        rs1_i    => rs1_i,
        rs2_i    => rs2_i,
        result_o => keccak_result_o,
        valid_o  => keccak_valid_o
      );
  end generate;

  -- NTT Engine (stdlib sub-unit; consumes funct3 = 110) ------------------------------------
  u_ntt: if CPU_CFU_NTT_EN generate
    ntt_engine: entity neorv32.neorv32_cpu_alu_cfu_ntt
      generic map (
        USE_DSP => CPU_CFU_NTT_USE_DSP
      )
      port map (
        clk_i    => clk_i,
        rstn_i   => rstn_i,
        start_i  => start_i,
        inst_i   => inst_i,
        rs1_i    => rs1_i,
        rs2_i    => rs2_i,
        result_o => ntt_result_o,
        valid_o  => ntt_valid_o
      );
  end generate;

  -- Rejection-Sampling Engine (stdlib sub-unit; consumes funct3 = 111) -----------------------
  rej_engine: entity neorv32.neorv32_cpu_alu_cfu_rej
    port map (
      clk_i    => clk_i,
      rstn_i   => rstn_i,
      start_i  => start_i,
      inst_i   => inst_i,
      rs1_i    => rs1_i,
      rs2_i    => rs2_i,
      result_o => rej_result_o,
      valid_o  => rej_valid_o
    );

  -- Response mux: the engine addressed by funct3 owns the CFU response ----------------------
  -- funct3 mapping:
  --   000..101 : Keccak-f1600 (6 opcodes)
  --   110      : NTT engine (8 sub-ops via funct7)
  --   111      : Rejection sampling (8 sub-ops via funct7)
  result_o <= ntt_result_o when (CPU_CFU_NTT_EN and funct3 = cfu_ntt_c) else
              rej_result_o when (funct3 = cfu_rej_c) else keccak_result_o;
  valid_o  <= ntt_valid_o  when (CPU_CFU_NTT_EN and funct3 = cfu_ntt_c) else
              rej_valid_o  when (funct3 = cfu_rej_c) else keccak_valid_o;

end neorv32_cpu_alu_cfu_rtl;
