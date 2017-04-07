'use strict';

const utils = require('./utils');
const dynamodb = require('./dynamodb');
const stats = require('./stats');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

/** Calculates the return for a company from a startDate to an endDate.  Adds
*    the capital gain and any dividends paid out and returns the percentage
   increase / decrease and the risk adjusted increase / decrease
* @param {Number} currentPrice - current day price
* @param {Number} purchasePrice - the original price
* @param {Number} dividends - the dividends for the period
* @param {Number} volatility - the price std deviation
* @return {Object} in the form of:
   {
   returns: 0.23, (percentage of purchase price )
   riskAdjustedReturns: 0.19 (percentage of purchase price divided by std dev)
 }

*/
let calculateReturnForPeriod = function(currentPrice, purchasePrice,
  dividends, volatility) {
  let priceReturn = (currentPrice - purchasePrice + dividends) / purchasePrice;
  let riskAdjustedReturns;

  // Convert to percentage
  priceReturn = priceReturn * 100;
  // Convert volatility to percentage
  volatility = volatility * 100;
  if (volatility < 1) {
    volatility = 1;
  }

  // For a positive return divide return by risk, for negative multiply by risk
  if (priceReturn >= 0) {
    riskAdjustedReturns = priceReturn / volatility;
  } else {
    riskAdjustedReturns = priceReturn * volatility;
  }

  return {
    returns: priceReturn,
    riskAdjustedReturns: riskAdjustedReturns,
  };
};

/** Calculates the return for a company from a startDate to an endDate.  Adds
*    the capital gain and any dividends paid out and returns the raw number
*    as well as the yield (percentage of the price at the start Date)
* @param {String} startDate - the quote date
* @param {Number} endDate - current day price
* @param {Array} dividends - dividends object in form:
  @return {Number} total dividends for period
*/
let returnDividendsForPeriod = asyncify(function(startDate, endDate,
  dividends) {
  let totalDividend = 0;
  Object.keys(dividends).forEach((dividendDate) => {
    if (dividendDate >= startDate && dividendDate <= endDate) {
      totalDividend += dividends[dividendDate];
    }
  });

  return totalDividend;
});

/** Updates the returns for previous periods using the prices and dividends.
   Updates total return for 1,2,4,8,12,26,52 weeks as well as risk adjusted
   return
* @param {String} symbol - the symbol to update
* @param {String} currentDate - the date for the current share price
* @param {Number} currentPrice - the current share price
* @param {Array} historicalValues - last 52 weeks of adjusted shared prices:
      [{"2007-01-03": 12.23}, {"2007-01-04": 12.25}] ...
* @param {Array} dividends - dividends paid in the last 52 weeks:
      [{"2007-02-20": 0.54}, {"2007-07-27": 0.56}] ...
*       {
*       'riskAdjustedReturn': 23.8977, (dollar amount)
*       'riskAdjustedYield': 0.23, (percentage of purchase price )
*/
let updateReturns = asyncify(function(symbol, currentDate, currentPrice,
  historicalValues, dividends) {
  // Set-up update details, only update the record if it is found
  let updateDetails = {
    tableName: 'companyQuotes',
    key: {
      symbol: symbol,
    },
    conditionExpression: 'attribute_exists(symbol) and ' +
      'attribute_exists(quoteDate)',
  };
  // Return arrays for prices by time period
  let weeklyStats = getWeeklyStats(historicalValues, currentDate);

  // Get prices and dividends
  let pricesAndDividends = {
    week1: {
      num: 1,
      quoteDate: weeklyStats['1WeekDate'],
      price: historicalValues[weeklyStats['1WeekDate']] || 0,
      stdDev: weeklyStats['1WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['1WeekDate'],
        currentDate, dividends)),
    },
    week2: {
      num: 2,
      quoteDate: weeklyStats['2WeekDate'],
      price: historicalValues[weeklyStats['2WeekDate']] || 0,
      stdDev: weeklyStats['2WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['2WeekDate'],
        currentDate, dividends)),
    },
    week4: {
      num: 4,
      quoteDate: weeklyStats['4WeekDate'],
      price: historicalValues[weeklyStats['4WeekDate']] || 0,
      stdDev: weeklyStats['4WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['4WeekDate'],
        currentDate, dividends)),
    },
    week8: {
      num: 8,
      quoteDate: weeklyStats['8WeekDate'],
      price: historicalValues[weeklyStats['8WeekDate']] || 0,
      stdDev: weeklyStats['8WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['8WeekDate'],
        currentDate, dividends)),
    },
    week12: {
      num: 12,
      quoteDate: weeklyStats['12WeekDate'],
      price: historicalValues[weeklyStats['12WeekDate']] || 0,
      stdDev: weeklyStats['12WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['12WeekDate'],
        currentDate, dividends)),
    },
    week26: {
      num: 26,
      quoteDate: weeklyStats['26WeekDate'],
      price: historicalValues[weeklyStats['26WeekDate']] || 0,
      stdDev: weeklyStats['26WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['26WeekDate'],
        currentDate, dividends)),
    },
    week52: {
      num: 52,
      quoteDate: weeklyStats['52WeekDate'],
      price: historicalValues[weeklyStats['52WeekDate']] || 0,
      stdDev: weeklyStats['52WeekStdDev'],
      dividends: awaitify(returnDividendsForPeriod(weeklyStats['52WeekDate'],
        currentDate, dividends)),
    },
  };

  // Calculate return and risk adjusted return for each period
  Object.keys(pricesAndDividends).forEach((week) => {
    let prefix = pricesAndDividends[week]['num'];
    let quoteDate = pricesAndDividends[week]['quoteDate'];
    let weekPrice = pricesAndDividends[week]['price'];
    let weekStd = pricesAndDividends[week]['stdDev'];
    let weekDividends = pricesAndDividends[week]['dividends'];

    if (weekPrice) {
      let weekReturns = calculateReturnForPeriod(currentPrice, weekPrice,
        weekDividends || 0, weekStd);

      updateDetails.key.quoteDate = quoteDate;
      updateDetails.updateExpression = 'set ' +
        '#Future' + prefix + 'WeekPrice = :' + prefix + 'WeekFuturePrice, ' +
        '#Future' + prefix + 'WeekDividend = :' +
        prefix + 'WeekFutureDividend, ' +
        '#Future' + prefix + 'WeekReturn = :' + prefix + 'WeekFutureReturn, ' +
        '#Future' + prefix + 'WeekRiskAdjustedReturn = :' +
        prefix + 'WeekFutureRiskAdjustedReturn';

      updateDetails.expressionAttributeNames = {};
      updateDetails.expressionAttributeNames['#Future' + prefix + 'WeekPrice'] = 'Future' + prefix + 'WeekPrice';
      updateDetails.expressionAttributeNames['#Future' + prefix + 'WeekDividend'] = 'Future' + prefix + 'WeekDividend';
      updateDetails.expressionAttributeNames['#Future' + prefix + 'WeekReturn'] = 'Future' + prefix + 'WeekReturn';
      updateDetails.expressionAttributeNames['#Future' + prefix + 'WeekRiskAdjustedReturn'] = 'Future' + prefix + 'WeekRiskAdjustedReturn';


      updateDetails.expressionAttributeValues = {};
      updateDetails.expressionAttributeValues[':' + prefix + 'WeekFuturePrice'] = currentPrice;
      updateDetails.expressionAttributeValues[':' + prefix + 'WeekFutureDividend'] = weekDividends || 0;
      updateDetails.expressionAttributeValues[':' + prefix + 'WeekFutureReturn'] = weekReturns.returns;
      updateDetails.expressionAttributeValues[':' + prefix + 'WeekFutureRiskAdjustedReturn'] = weekReturns.riskAdjustedReturns;

      try {
        awaitify(dynamodb.updateRecord(updateDetails));
      } catch (err) {
        console.log(err);
      }
    }
  });
});


/** Returns the average price and std deviation for an array of prices for the
  previous 1, 2, 4, 8, 12, 26, 52 weeks of values
* @param {Array} historicalValues - the complete set of values in the form:
  [{"2007-01-03": 12.23}, {"2007-01-04": 12.25}] ...
* @param {String} currentDate - the date to use to calculate the division
* @return {Object} a series of arrays aplit into time periods:
*      {
        "1Week": 12.456,
        "2Week": 12.765,
        "4Week": 11.124,
        ...
        }
*/
let getWeeklyStats = function(historicalValues, currentDate) {
  // Return arrays for prices by time period
  let weekPrices = splitPricesIntoPeriods(historicalValues, currentDate);
  let returnVals = {};

  returnVals['1WeekDate'] = weekPrices['1WeekDate'];
  returnVals['2WeekDate'] = weekPrices['2WeekDate'];
  returnVals['4WeekDate'] = weekPrices['4WeekDate'];
  returnVals['8WeekDate'] = weekPrices['8WeekDate'];
  returnVals['12WeekDate'] = weekPrices['12WeekDate'];
  returnVals['26WeekDate'] = weekPrices['26WeekDate'];
  returnVals['52WeekDate'] = weekPrices['52WeekDate'];

  returnVals['1WeekAverage'] = stats.average(weekPrices['1WeekVals']);
  returnVals['2WeekAverage'] = stats.average(weekPrices['2WeekVals']);
  returnVals['4WeekAverage'] = stats.average(weekPrices['4WeekVals']);
  returnVals['8WeekAverage'] = stats.average(weekPrices['8WeekVals']);
  returnVals['12WeekAverage'] = stats.average(weekPrices['12WeekVals']);
  returnVals['26WeekAverage'] = stats.average(weekPrices['26WeekVals']);
  returnVals['52WeekAverage'] = stats.average(weekPrices['52WeekVals']);

  // Std deviation - volatility
  returnVals['1WeekStdDev'] = stats.stdDev(weekPrices['1WeekVals']);
  returnVals['2WeekStdDev'] = stats.stdDev(weekPrices['2WeekVals']);
  returnVals['4WeekStdDev'] = stats.stdDev(weekPrices['4WeekVals']);
  returnVals['8WeekStdDev'] = stats.stdDev(weekPrices['8WeekVals']);
  returnVals['12WeekStdDev'] = stats.stdDev(weekPrices['12WeekVals']);
  returnVals['26WeekStdDev'] = stats.stdDev(weekPrices['26WeekVals']);
  returnVals['52WeekStdDev'] = stats.stdDev(weekPrices['52WeekVals']);

  // High and low
  returnVals['52WeekHigh'] = stats.max(weekPrices['52WeekVals']);
  returnVals['52WeekLow'] = stats.min(weekPrices['52WeekVals']);

  // Bollinger bands - 4 and 12 weeks
  returnVals['4WeekBollingerBandUpper'] = returnVals['4WeekAverage'] +
    (2 * returnVals['4WeekStdDev']);
  returnVals['4WeekBollingerBandLower'] = returnVals['4WeekAverage'] -
    (2 * returnVals['4WeekStdDev']);

  returnVals['12WeekBollingerBandUpper'] = returnVals['12WeekAverage'] +
    (2 * returnVals['12WeekStdDev']);
  returnVals['12WeekBollingerBandLower'] = returnVals['12WeekAverage'] -
    (2 * returnVals['12WeekStdDev']);

  return returnVals;
};

/** Splits an array with time values into separate arrays wih 1, 2, 4, 8, 12,
    26, 52 weeks of values
* @param {Array} values - the complete set of values in the form:
  [{"2007-01-03": 12.23}, {"2007-01-04": 12.25}] ...
* @param {String} currentDate - the date to use to calculate the division
* @return {Object} a series of arrays aplit into time periods:
*      {
        "1WeekDate" :'2016-04-06',
        ...
        "1WeekVals": [12.23,12.24,12.22,12,25,12,24],
        "2WeekVals": [12.23,12.24,12.22,12,25,12,24.....]
        "4WeekVals": [12.23,12.24,12.22,12,25,12,24.....],
        ...
        }
*/
let splitPricesIntoPeriods = function(values, currentDate) {
  if (!utils.isDate(currentDate)) {
    console.error(`splitPricesIntoPeriods contains invalid date parameter`,
      `values: ${JSON.stringify(values)}, currentDate: ${currentDate}`);
    return;
  }

  let allVals = {
    '1WeekDate': utils.dateAdd(currentDate, 'weeks', -1),
    '2WeekDate': utils.dateAdd(currentDate, 'weeks', -2),
    '4WeekDate': utils.dateAdd(currentDate, 'weeks', -4),
    '8WeekDate': utils.dateAdd(currentDate, 'weeks', -8),
    '12WeekDate': utils.dateAdd(currentDate, 'weeks', -12),
    '26WeekDate': utils.dateAdd(currentDate, 'weeks', -26),
    '52WeekDate': utils.dateAdd(currentDate, 'weeks', -52),
    '1WeekVals': [],
    '2WeekVals': [],
    '4WeekVals': [],
    '8WeekVals': [],
    '12WeekVals': [],
    '26WeekVals': [],
    '52WeekVals': [],
  };

  Object.keys(values).forEach((valueDate) => {
    // Check value date is before current date, if not, it should be ignored
    if (valueDate <= currentDate) {
      // For each reference date check if this value can be used
      if (valueDate > allVals['1WeekDate']) {
        allVals['1WeekVals'].push(values[valueDate]);
      }
      if (valueDate > allVals['2WeekDate']) {
        allVals['2WeekVals'].push(values[valueDate]);
      }
      if (valueDate > allVals['4WeekDate']) {
        allVals['4WeekVals'].push(values[valueDate]);
      }
      if (valueDate > allVals['8WeekDate']) {
        allVals['8WeekVals'].push(values[valueDate]);
      }
      if (valueDate > allVals['12WeekDate']) {
        allVals['12WeekVals'].push(values[valueDate]);
      }
      if (valueDate > allVals['26WeekDate']) {
        allVals['26WeekVals'].push(values[valueDate]);
      }
      if (valueDate > allVals['52WeekDate']) {
        allVals['52WeekVals'].push(values[valueDate]);
      }
    }
  });

  return allVals;
};

module.exports = {
  getWeeklyStats: getWeeklyStats,
  updateReturns: updateReturns,
};
