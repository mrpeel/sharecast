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