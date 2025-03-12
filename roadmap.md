# Project Roadmap

## ✅ Completed

- fetch data, start with daily or 8h klines to remove nuance and abstract generalities
  - get raw data
- clean data
  - clean raw data or handle edge cases
- visualize data, what does the market look like?
  - start with most natural window size, 8h
  - visualize raw data (to ensure cleaning is not obviously off)
  - visualize cleaned data (to develop intuition and understanding and start to be curious about how the market functions)
  - limited analysis (to extend intuition and develop questions to request more data or refine existing data)
  - dialate data to 1d
  - contract data to 1h
  - contract to 1m
  - check out other indices (ETH, sUSDe)
- ask the data questions, be curious about why things might look the way they do
  -consider edges, scenarios, and regimes.
  -consider business case of the business of trading.

- describe the point of the algo
- use words to describe a naive model, explain it to my cat
  - implement model
  - review output (visually and backtest)
  - think about it
- iterate with more granular data

## 🚧 In Progress

## 🎯 Next Steps

## 🔄 Notes & Considerations

- Binance funding rates are given every 8 hours – convert to APR
- Need to handle contract expirations for quarterly futures
- Structuring data cleanly will help with later analytics & modeling

### Links

Funding rates in crypto markets

- https://www.binance.com/en/square/post/11952138240353

Introduction to Binance Futures Funding Rates

- https://www.binance.com/en/support/faq/detail/360033525031

BTCUSDT Perp Premium Index

- https://www.binance.com/en/futures/funding-history/perpetual/index

What are Mark Price and Price Index?

- https://www.binance.com/en/support/faq/detail/360033525071

How funding works on Binance

- https://fsr-develop.com/blog-cscalp/tpost/r48x284bt1-binance-funding-rate-how-funding-works-o

- https://www.binance.com/en/blog/futures/what-is-futures-funding-rate-and-why-it-matters-421499824684903247

- https://stackoverflow.com/questions/70178955/how-can-i-obtain-this-specific-series-data-to-calculate-time-to-funding-weighted
  #"Average Premium Index (P) = (1 _ Premium_Index_1 + 2 _ Premium_Index_2 + 3 _ Premium_Index_3 +···+·480 _ Premium_index_480)/(1+2+3+···+480)"
