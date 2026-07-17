# Market snapshot provenance

This directory contains a fixed research snapshot for the V1 paper-trading universe. It is
included so the example backtest can run without contacting a market-data provider.

- Provider: Sina Finance
- Raw daily bars: `https://stock.finance.sina.com.cn/usstock/api/jsonp.php/.../US_MinKService.getDailyK`
- Forward-adjustment factors: `https://finance.sina.com.cn/us_stock/company/reinstatement/{SYMBOL}_qfq.js`
- Adjustment: `adjusted_price = raw_price * qfq_factor + adjust`; volume is unchanged
- Requested interval: `[2023-01-01, 2026-01-01)`
- Actual coverage: 2023-01-03 through 2025-12-31
- Retrieved: 2026-07-17 15:19–15:20 UTC
- Rows: 752 per symbol
- Symbols: SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, GOOGL, META

Each `data/market/TICKER.json` manifest records the immutable Parquet filename, row count,
retrieval timestamp, maximum market date, and SHA-256 digest. `ParquetMarketCache.read()` verifies
the digest and validates the full OHLCV frame before returning it.

| Symbol | Parquet SHA-256 |
| --- | --- |
| SPY | `ccbccc4c644ee80a074e728a604306011e9c17144a5bf2ce0bade3949b66725a` |
| QQQ | `6b3fa10fc3173c0038ce6910730fa57620bd20e37ba8973b2614720dee31be09` |
| IWM | `8cf86fd701b465f9b96ea56b72e33f2101ecc7bab30c6d0862217ad4906727a5` |
| AAPL | `bd567d645eb9f9bf9aa42890b2188cfbb742ae5855dd7ea90c4c6a3b996aa99f` |
| MSFT | `021955ebb50ca556a3af81006a9ff853988cfb87f90abc7d333ba69c1e3c6597` |
| NVDA | `e51f039a2d1de36f8c29dc3ecb1085bc7c6b4933ba5b3434707263e2fd77e1e8` |
| AMZN | `4121ec5660b2e2321af370007a2697ba77574f670c3e50babcffb9f89c8384be` |
| GOOGL | `cf652c559325588e5a661f253da39c8c69853c620da255578636a3ccddf34b0e` |
| META | `31c310e31d3f929afe0628e28089dc64583778b9bb96813bf2668b2df2ed9755` |

The provider can change or remove these endpoints. This snapshot is for research and paper
simulation only, is not investment advice, and should not be treated as a licensed real-time feed.
