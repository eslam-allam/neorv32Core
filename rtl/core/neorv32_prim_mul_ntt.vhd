-- ================================================================================ --
-- NEORV32 Primitives - Generic Signed Integer Multiplier for NTT (32x27)         --
-- -------------------------------------------------------------------------------- --
-- Combinational multiplier: the NTT FSM issues one multiply per cycle in the       --
-- S_M1/S_M2/S_M3 states and consumes the product on the next clock edge, so the    --
-- primitive must not add latency. (A registered multiplier would shift the whole   --
-- butterfly chain by its pipeline depth.)                                          --
--                                                                                  --
-- The B operand is always 27 bits (signed), which fits DSP48E1's 18x27 natively.  --
-- USE_DSP = true  : instantiate DSP48E1 primitives (Xilinx 7-series)              --
-- USE_DSP = false : behavioral multiplier (tool decides: LUTs or inferred DSP)    --
-- ================================================================================ --

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neorv32_prim_mul_ntt is
  generic (
    USE_DSP : boolean := false  -- true = use DSP48E1 primitive (Xilinx 7-series)
  );
  port (
    -- data path --
    a_i     : in  signed(31 downto 0);             -- operand A (32-bit signed)
    b_i     : in  signed(26 downto 0);             -- operand B (27-bit signed)
    res_o   : out signed(58 downto 0)              -- 32+27-1 = 58 bits product
  );
end neorv32_prim_mul_ntt;

architecture neorv32_prim_mul_ntt_rtl of neorv32_prim_mul_ntt is

  component DSP48E1
    generic (
      ACASCREG           : integer := 1;
      ADREG              : integer := 1;
      ALUMODEREG         : integer := 1;
      AREG               : integer := 1;
      AUTORESET_PATDET   : string  := "NO_RESET";
      A_INPUT            : string  := "DIRECT";
      BCASCREG           : integer := 1;
      BREG               : integer := 1;
      B_INPUT            : string  := "DIRECT";
      CARRYINREG         : integer := 1;
      CARRYINSELREG      : integer := 1;
      CREG               : integer := 1;
      DREG               : integer := 1;
      INMODEREG          : integer := 1;
      MASK               : bit_vector := X"3FFFFFFFFFFF";
      MREG               : integer := 1;
      OPMODEREG          : integer := 1;
      PATTERN            : bit_vector := X"000000000000";
      PREG               : integer := 1;
      SEL_MASK           : string  := "MASK";
      SEL_PATTERN        : string  := "PATTERN";
      -- NOTE: USE_DPORT is a string (not the boolean used by the VHDL
      -- template) so the mixed-language xsim flow passes "FALSE" to the
      -- Verilog unisim model; synthesis accepts the same value.
      USE_DPORT          : string  := "FALSE";
      USE_MULT           : string  := "MULTIPLY";
      USE_PATTERN_DETECT : string  := "NO_PATDET";
      USE_SIMD           : string  := "ONE48"
    );
    port (
      ACOUT          : out std_logic_vector(29 downto 0);
      BCOUT          : out std_logic_vector(17 downto 0);
      CARRYCASCOUT   : out std_logic;
      CARRYOUT       : out std_logic_vector(3 downto 0);
      MULTSIGNOUT    : out std_logic;
      OVERFLOW       : out std_logic;
      P              : out std_logic_vector(47 downto 0);
      PATTERNBDETECT : out std_logic;
      PATTERNDETECT  : out std_logic;
      PCOUT          : out std_logic_vector(47 downto 0);
      UNDERFLOW      : out std_logic;
      A              : in  std_logic_vector(29 downto 0);
      ACIN           : in  std_logic_vector(29 downto 0);
      ALUMODE        : in  std_logic_vector(3 downto 0);
      B              : in  std_logic_vector(17 downto 0);
      BCIN           : in  std_logic_vector(17 downto 0);
      C              : in  std_logic_vector(47 downto 0);
      CARRYCASCIN    : in  std_logic;
      CARRYIN        : in  std_logic;
      CARRYINSEL     : in  std_logic_vector(2 downto 0);
      CEA1           : in  std_logic;
      CEA2           : in  std_logic;
      CEAD           : in  std_logic;
      CEALUMODE      : in  std_logic;
      CEB1           : in  std_logic;
      CEB2           : in  std_logic;
      CEC            : in  std_logic;
      CECARRYIN      : in  std_logic;
      CECTRL         : in  std_logic;
      CED            : in  std_logic;
      CEINMODE       : in  std_logic;
      CEM            : in  std_logic;
      CEP            : in  std_logic;
      CLK            : in  std_logic;
      D              : in  std_logic_vector(24 downto 0);
      INMODE         : in  std_logic_vector(4 downto 0);
      MULTSIGNIN     : in  std_logic;
      OPMODE         : in  std_logic_vector(6 downto 0);
      PCIN           : in  std_logic_vector(47 downto 0);
      RSTA           : in  std_logic;
      RSTALLCARRYIN  : in  std_logic;
      RSTALUMODE     : in  std_logic;
      RSTB           : in  std_logic;
      RSTC           : in  std_logic;
      RSTCTRL        : in  std_logic;
      RSTD           : in  std_logic;
      RSTINMODE      : in  std_logic;
      RSTM           : in  std_logic;
      RSTP           : in  std_logic
    );
  end component;

  -- operand slices (low parts are zero-extended = unsigned, high parts are signed)
  type a_arr_t is array (0 to 3) of signed(24 downto 0);
  type b_arr_t is array (0 to 3) of signed(17 downto 0);
  type p_arr_t is array (0 to 3) of std_logic_vector(47 downto 0);

  signal a_lo, a_hi : signed(24 downto 0);
  signal b_lo, b_hi : signed(17 downto 0);
  signal op_a       : a_arr_t;
  signal op_b       : b_arr_t;
  signal dsp_p      : p_arr_t;
  signal sum        : signed(58 downto 0);

begin

  -- ------------------------------------------------------------------
  -- Fabric / tool-decides implementation (combinational)
  -- ------------------------------------------------------------------
  gen_lut: if not USE_DSP generate
    signal prod : signed(58 downto 0);
    attribute use_dsp : string;
    attribute use_dsp of prod : signal is "no";
  begin
    prod  <= a_i * b_i;
    res_o <= prod;
  end generate;

  -- ------------------------------------------------------------------
  -- Forced DSP48E1 implementation: 4 DSPs + fabric sum
  -- All pipeline registers are bypassed (AREG/BREG/MREG/PREG=0) so the
  -- result is available in the same cycle, like the LUT path.
  -- ------------------------------------------------------------------
  gen_dsp: if USE_DSP generate

    -- operand split
    a_lo <= signed(resize(unsigned(a_i(23 downto 0)), 25));
    a_hi <= resize(a_i(31 downto 24), 25);
    b_lo <= signed(resize(unsigned(b_i(16 downto 0)), 18));
    b_hi <= resize(b_i(26 downto 17), 18);

    op_a(0) <= a_lo;  op_b(0) <= b_lo;   -- weight 2^0
    op_a(1) <= a_lo;  op_b(1) <= b_hi;   -- weight 2^17
    op_a(2) <= a_hi;  op_b(2) <= b_lo;   -- weight 2^24
    op_a(3) <= a_hi;  op_b(3) <= b_hi;   -- weight 2^41

    gen_mul: for i in 0 to 3 generate
      dsp_inst: DSP48E1
        generic map (
          ACASCREG      => 0,           -- must match AREG=0
          ADREG         => 0,
          ALUMODEREG    => 0,
          AREG          => 0,
          BCASCREG      => 0,           -- must match BREG=0
          BREG          => 0,
          CARRYINREG    => 0,
          CARRYINSELREG => 0,
          CREG          => 0,
          DREG          => 0,
          INMODEREG     => 0,
          MREG          => 0,
          OPMODEREG     => 0,
          PREG          => 0,
          USE_DPORT     => "FALSE",
          USE_MULT      => "MULTIPLY",
          USE_SIMD      => "ONE48"
        )
        port map (
          A              => std_logic_vector(resize(op_a(i), 30)),
          ACIN           => (others => '0'),
          ALUMODE        => "0000",
          B              => std_logic_vector(op_b(i)),
          BCIN           => (others => '0'),
          C              => (others => '0'),
          CARRYCASCIN    => '0',
          CARRYIN        => '0',
          CARRYINSEL     => "000",
          CEA1           => '0',
          CEA2           => '0',
          CEAD           => '0',
          CEALUMODE      => '0',
          CEB1           => '0',
          CEB2           => '0',
          CEC            => '0',
          CECARRYIN      => '0',
          CECTRL         => '0',
          CED            => '0',
          CEINMODE       => '0',
          CEM            => '0',
          CEP            => '0',
          CLK            => '0',         -- no internal registers
          D              => (others => '0'),
          INMODE         => "00000",    -- A * B
          MULTSIGNIN     => '0',
          OPMODE         => "0000101",  -- Z=0, Y=M, X=M  ->  P = A*B
          PCIN           => (others => '0'),
          RSTA           => '0',
          RSTALLCARRYIN  => '0',
          RSTALUMODE     => '0',
          RSTB           => '0',
          RSTC           => '0',
          RSTCTRL        => '0',
          RSTD           => '0',
          RSTINMODE      => '0',
          RSTM           => '0',
          RSTP           => '0',
          ACOUT          => open,
          BCOUT          => open,
          CARRYCASCOUT   => open,
          CARRYOUT       => open,
          MULTSIGNOUT    => open,
          OVERFLOW       => open,
          P              => dsp_p(i),
          PATTERNBDETECT => open,
          PATTERNDETECT  => open,
          PCOUT          => open,
          UNDERFLOW      => open
        );
    end generate;

    -- recombine partial products
    sum <= resize(signed(dsp_p(0)), 59)
         + shift_left(resize(signed(dsp_p(1)), 59), 17)
         + shift_left(resize(signed(dsp_p(2)), 59), 24)
         + shift_left(resize(signed(dsp_p(3)), 59), 41);

    res_o <= sum;

  end generate;

end neorv32_prim_mul_ntt_rtl;
