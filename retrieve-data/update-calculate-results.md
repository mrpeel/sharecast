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
 - calculate dividend % : IAD (annual dividend) / 52 * number of weeks in the future
 - add together using a double weighting for dividends (dividends are more important)


  Recommendation calculation:
    (FuturePrice - price for week) / (price for the week) +
    (price for the week) / dividend calculation * 2):
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
