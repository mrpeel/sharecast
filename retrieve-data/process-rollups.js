'use strict';

const utils = require('./utils');
const dynamodb = require('./dynamodb');
const stats = require('./stats');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

/** Caclulates and updates the volatility information for a company quote
* @param {String} symbol - the company symbol
* @param {String} quoteDate - the quote date
*/
let calculateVolatility = asyncify(function(symbol, quoteDate, currentPrice) {
  // Set-up week differences: equates to 1,2,4,8,12,26,52
  let diffWeeks = {
    '1WeekVolatility': -1,
    '2WeekVolatility': -1,
    '4WeekVolatility': -2,
    '8WeekVolatility': -4,
    '12WeekVolatility': -4,
    '26WeekVolatility': -14,
    '52WeekVolatility': -26,
  };
  let quoteVals = [];
  let endDate = quoteDate;
  let startDate;

  Object.keys(diffWeeks).forEach((volatility) => {
    // Set start date
    startDate = utils.dateAdd(endDate, 'weeks', diffWeeks[volatility]);
    // Retrieve values
    quoteVals = awaitify(getQuoteVals(quoteVals, startDate, endDate));

    let std = stats.standardDeviation(quoteVals);

    let volatilityVal = std / currentPrice;

    awaitify(setQuoteVal(symbol, quoteDate, volatility, volatilityVal));

    // Reset end date for next run
    endDate = startDate;
  });
});

/** Get quote values and appends them to an existing array
* @param {Array} quoteVals - the array to append results to
* @param {String} startDate - the time period start date
* @param {String} endDate - the time period end date
* @return {Array} updated array with new values from query results
*/
let getQuoteVals = asyncify(function(quoteVals, startDate, endDate) {
  let queryDetails = {};
  let queryResult = awaitify(dynamodb.queryTable());

  if (queryResult.length) {
    quoteVals.push();
  }

  return quoteVals;
});

/** Sets a company quote value item
* @param {Array} symbol - the array to append results to
* @param {String} quoteDate - the time period start date
* @param {Array} attributeNameValues - an array of values to set:
*           [{
              AttributeName: '1WeekVolatility',
              AttributeValue: 0.000123457,
            },
            ]
*/
let setQuoteVal = asyncify(function(symbol, quoteDate,
  attributeNameValues) {
  let updateDetails = {};

  // Loop through values and prepare set statement

  awaitify(dynamodb.updateRecord());

  if (queryResult.length) {
    quoteVals.push();
  }
});

/** Caclulates the price change for the previous day and previous 52 week high
*     and low value
* @param {String} symbol - the company symbol
* @param {String} quoteDate - the quote date
* @param {Number} currentPrice - current day price
*/
let calculatePriceChanges = asyncify(function(symbol, quoteDate,
  currentPrice) {
  //

});

/** Determines whether dividencs were paid during time period, and, if paid,
*   returns them
* @param {String} symbol - the company symbol
* @param {String} startDate - the quote date
* @param {Number} endDate - current day price
* @return {Number} dividends paid
*/
let returnDividendsPaidForPeriod = asyncify(function(symbol, startDate,
  endDate) {
  //

});

/** Calculates the return for a company from a startDate to an endDate.  Adds
*    the capital gain and any dividends paid out and returns the raw number
*    as well as the yield (percentage of the price at the start Date)
* @param {String} symbol - the company symbol
* @param {String} startDate - the quote date
* @param {Number} endDate - current day price
* @param {Number} currentPrice - current day price
* @return {Object} in form of:
*       {
*       'totalReturn': 23.8977, (dollar amount)
*       'totalYield': 0.23, (percentage of purchase price )
*/
let calculateTotalReturnForPeriod = asyncify(function(symbol, startDate,
  endDate, currenPrice) {
  //

});

/** Calculates the risk adjusted return for a time period as a return and
*    as a yield of the purchase price.
*    Formula return: total return / std deviation
*    Formular percentage: purchase price / total return / std deviation
* @param {Number} purchasePrice - purchase price
* @param {Number} totalReturn - return for time period
* @param {Number} stdDeviation - the standard deviation for the time period
* @return {Object} in form of:
*       {
*       'riskAdjustedReturn': 23.8977, (dollar amount)
*       'riskAdjustedYield': 0.23, (percentage of purchase price )
*/
let calculateRiskAdjustedReturnForPeriod = asyncify(function(purchasePrice,
  totalReturn, stdDeviation) {
  //

});
