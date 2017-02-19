Updates to previous week values
----------
1 week ago -> update 1WeekFuturePrice: lastTradePriceOnly
2 weeks ago -> update 2WeekFuturePrice: lastTradePriceOnly
4 weeks ago -> update 4WeekFuturePrice: lastTradePriceOnly
8 weeks ago -> update 8WeekFuturePrice: lastTradePriceOnly
12 weeks ago -> update 12WeekFuturePrice: lastTradePriceOnly
26 weeks ago -> update 26WeekFuturePrice: lastTradePriceOnly
52 weeks ago -> update 52WeekFuturePrice: lastTradePriceOnly


Apply formula to each result:
- calculate capital gain % : future price / current price
- dividends not calculated for 1,2,4,8 weeks because the odds are they won't be paid
  during the period
- some values have DividendYield - use DividendYield / 52 * number of weeks in the future
OR
  - calculate dividend yield % : lastTradePriceOnly / DPSRecentYear / 52 * number of weeks in the future
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


  Recommendation calculation (weighted for dividensa over capital gain):
  1,2,4,8 weeks: CaptialGain
  12,26,52: CaptialGain + (Dividends * 2)

    if >= 0.05: buy
    if >= 0 and < 0.05: hold
    if < 0: sell
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
