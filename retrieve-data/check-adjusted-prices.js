'use strict';

const utils = require('./utils');
const dynamodb = require('./dynamodb');
const retrieveData = require('./dynamo-retrieve-share-data');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

/** Retrieve daily history values for symbol(s) from yahoo finance
* @param {Array} symbol - one or more yahoo symbols to lookup
* @param {String} startDate - the first day of the lookup period
* @param {String} endDate - the last day of the lookup period
* @return {Promise} resolves with the history record in form:
*    {
*    date: 20 Jan 2017,
*    open: 28.77,
*    high: 28.84,
*    low: 28.42,
*    close: 28.81,
*    volume: 536800,
*    adjClose: 27.73,
*    symbol: 'JBH.AX'
* }
*/
let retrieveDailyHistory = function(symbol, startDate, endDate) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      period: 'd',
    };

    // Check if one or many symbols
    if (Array.isArray(symbol)) {
      historyOptions.symbols = symbol;
    } else {
      historyOptions.symbol = symbol;
    }

    yahooFinance.historical(historyOptions).then(function(result) {
      resolve(result);
    }).catch(function(err) {
      reject(err);
    });
  });
};

/** Check through each company symbol, retrieve the last 2 weeks history data
*    and see if any prices have closePrice !== adjustedPrice.  If a discrepancy
*    is found, then all prices for that symbol need to be re-checked back to
*    2006-07-01, so trigger a complete re-check of prices.
*
*/
let checkForAdjustments = asyncify(function() {
  let endDate = utils.returnDateAsString(Date.now());
  let startDate = utils.dateAdd(endDate, 'weeks', -2);
  let symbolResult = awaitify(retrieveData.setupSymbols());

  let companies = symbolResult.companies;

  for (let companyCounter = startRec; companyCounter < endRec;
    companyCounter += 15) {
    symbolGroups.push(companies.slice(companyCounter, companyCounter + 15));
  }

  symbolGroups.forEach((symbolGroup) => {
    try {
      let result = awaitify(retrieveDailyHistory(symbolGroup,
        companyFieldsToRetrieve));

      result.forEach((historyRecord) => {
        // Check whether adjClose is different to close
      });
    } catch (err) {
      console.error(err);
    }
  });
});
