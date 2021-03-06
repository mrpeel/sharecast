Weekly aggregation of quote fields
----------

symbol -> key
previousClose -> min
dividendYield -> median
dividendPerShare -> median
exDividendDate -> median
change -> discard
changeInPercent -> discard
lastTradeDate -> discard
daysLow -> min
daysHigh -> max
lastTradePriceOnly -> median
lastTradePriceOnly -> std deviation (volatility)
52WeekHigh -> discard
52WeekLow -> discard
marketCapitalization -> median
sharesOwned -> median
stockExchange -> discard
volume -> sum
earningsPerShare -> median
epsEstimateCurrentYear -> median
bookValue -> median
ebitda -> median
pricePerSales -> median
pricePerBook -> median
peRatio -> median
pegRatio -> median
pricePerEpsEstimateCurrentYear -> median
pricePerEpsEstimateNextYear -> median
shortRatio -> discard

Further time period rollup
----------
2 weeks: lastTradePriceOnly -> std deviation (volatility)
4 weeks: lastTradePriceOnly -> std deviation (volatility)
8 weeks: lastTradePriceOnly -> std deviation (volatility)
12 weeks: lastTradePriceOnly -> std deviation (volatility)
26 weeks: lastTradePriceOnly -> std deviation (volatility)
52 weeks: lastTradePriceOnly -> std deviation (volatility)

Week to previous week comparisons
----------
1 week ago: sum of volume (across week) -> std deviation (volatility)
2 weeks ago: sum of volume (across week) -> std deviation (volatility)
4 weeks ago: sum of volume (across week) -> std deviation (volatility)
8 weeks ago: sum of volume (across week) -> std deviation (volatility)
12 weeks ago: sum of volume (across week) -> std deviation (volatility)
26 weeks ago: sum of volume (across week) -> std deviation (volatility)
52 weeks ago: sum of volume (across week) -> std deviation (volatility)

Updates to previous week roll-ups
----------
1 week ago -> update 1WeekFuturePrice: median lastTradePriceOnly (across week)
2 weeks ago -> update 2WeekFuturePrice: median lastTradePriceOnly (across week)
4 weeks ago -> update 4WeekFuturePrice: median lastTradePriceOnly (across week)
8 weeks ago -> update 8WeekFuturePrice: median lastTradePriceOnly (across week)
12 weeks ago -> update 12WeekFuturePrice: median lastTradePriceOnly (across week)
26 weeks ago -> update 26WeekFuturePrice: median lastTradePriceOnly (across week)
52 weeks ago -> update 52WeekFuturePrice: median lastTradePriceOnly (across week)


Apply formula to each result
- calculate capital gain % : future price / current price
- dividends not calculated for 1,2,4,8 weeks because the odds are they won't be paid
  during the period
- calculate dividend % : lastTradePriceOnly / dividendPerShare / 52 * number of weeks in the future
- add together using a double weighting for dividends (dividends are more important)

1 week ago -> update 1WeekFutureCaptialGain
2 weeks ago -> update 2WeekFutureCaptialGain
4 weeks ago -> update 4WeekFutureCaptialGain
8 weeks ago -> update 8WeekFutureCaptialGain
12 weeks ago -> update 12WeekFutureCaptialGain
26 weeks ago -> update 26WeekFutureCaptialGain
52 weeks ago -> update 52WeekFutureCaptialGain

12 weeks ago -> update 12WeekFutureDividendGain
26 weeks ago -> update 26WeekFutureDividendGain
52 weeks ago -> update 52WeekFutureDividendGain



  Recommendation calculation:
    1,2,4,8 weeks: CaptialGain
    12,26,52: CaptialGain + (Dividends * 2)

      if >= 0.05: buy
      if >= 0 and < 0.05: hold
      if < 0: sell


      1 week ago -> update 1WeekFutureRecommendation
      2 weeks ago -> update 2WeekFutureRecommendation
      4 weeks ago -> update 4WeekFutureRecommendation
      8 weeks ago -> update 8WeekFutureRecommendation
      12 weeks ago -> update 12WeekFutureRecommendation
      26 weeks ago -> update 26WeekFutureRecommendation
      52 weeks ago -> update 52WeekFutureRecommendation
