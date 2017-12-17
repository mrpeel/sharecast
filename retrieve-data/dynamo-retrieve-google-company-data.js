'use strict';

const fetch = require('node-fetch');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const dynamoSymbols = require('./dynamo-symbols');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

let baseUrl = 'https://finance.google.com/finance?output=json&start=0&num=5000&noIL=1&q=[currency%20%3D%3D%20%22AUD%22%20%26%20%28exchange%20%3D%3D%20%22ASX%22%29%20%26%20%28';
let suffixUrl = ']&restype=company&ei=X6iZWMmFIMGW0AThhLPoCw';


let fields = {
  'base': 'market_cap%20%3E%3D%2010000000%29%20%26%20%28market_cap%20%3C%3D%2014039000000000%29' +
    '%20%26%20%28last_price%20%3E%3D%200%29%20%26%20%28last_price%20%3C%3D%2030400%29',
  'agg1': 'earnings_per_share%20%3E%3D%20-8121%29%20%26%20%28earnings_per_share%20%3C%3D%203679%29' +
    '%20%26%20%28average_200day_price%20%3E%3D%20-1000%29%20%26%20%28average_200day_price%20%3C%3D%20199000%29' +
    '%20%26%20%28price_change_52week%20%3E%3D%20-9947%29%20%26%20%28price_change_52week%20%3C%3D%201420100%29' +
    '%20%26%20%28dividend_recent_quarter%20%3E%3D%200%29%20%26%20%28dividend_recent_quarter%20%3C%3D%20913%29' +
    '%20%26%20%28price_to_book%20%3E%3D%200%29%20%26%20%28price_to_book%20%3C%3D%20266400%29',
  'agg2': 'pe_ratio%20%3E%3D%200%29%20%26%20%28pe_ratio%20%3C%3D%201630100%29' +
    '%20%26%20%28dividend_per_share_trailing_12months%20%3E%3D%200%29%20%26%20%28dividend_per_share_trailing_12months%20%3C%3D%20913%29' +
    '%20%26%20%28dividend_next_quarter%20%3E%3D%200%29%20%26%20%28dividend_next_quarter%20%3C%3D%20223%29' +
    '%20%26%20%28dividend_per_share%20%3E%3D%200%29%20%26%20%28dividend_per_share%20%3C%3D%20913%29' +
    '%20%26%20%28dividend_next_year%20%3E%3D%200%29%20%26%20%28dividend_next_year%20%3C%3D%20598%29',
  'agg3': 'dividend_yield%20%3E%3D%200%29%20%26%20%28dividend_yield%20%3C%3D%2075100%29' +
    '%20%26%20%28dividend_recent_year%20%3E%3D%200%29%20%26%20%28dividend_recent_year%20%3C%3D%20582800%29' +
    '%20%26%20%28book_value_per_share_year%20%3E%3D%20-2298%29%20%26%20%28book_value_per_share_year%20%3C%3D%2010100%29' +
    '%20%26%20%28cash_per_share_year%20%3E%3D%200%29%20%26%20%28cash_per_share_year%20%3C%3D%206587%29' +
    '%20%26%20%28current_assets_to_liabilities_ratio_year%20%3E%3D%200%29%20%26%20%28current_assets_to_liabilities_ratio_year%20%3C%3D%206923200%29',
  'agg4': 'longterm_debt_to_assets_year%20%3E%3D%200%29%20%26%20%28longterm_debt_to_assets_year%20%3C%3D%20102000000%29' +
    '%20%26%20%28longterm_debt_to_assets_quarter%20%3E%3D%200%29%20%26%20%28longterm_debt_to_assets_quarter%20%3C%3D%20102000000%29' +
    '%20%26%20%28total_debt_to_assets_year%20%3E%3D%200%29%20%26%20%28total_debt_to_assets_year%20%3C%3D%20208999999.99999998%29' +
    '%20%26%20%28total_debt_to_assets_quarter%20%3E%3D%200%29%20%26%20%28total_debt_to_assets_quarter%20%3C%3D%20208999999.99999998%29' +
    '%20%26%20%28longterm_debt_to_equity_year%20%3E%3D%200%29%20%26%20%28longterm_debt_to_equity_year%20%3C%3D%202892500%29',
  'agg5': 'longterm_debt_to_equity_quarter%20%3E%3D%200%29%20%26%20%28longterm_debt_to_equity_quarter%20%3C%3D%202892500%29' +
    '%20%26%20%28total_debt_to_equity_year%20%3E%3D%200%29%20%26%20%28total_debt_to_equity_year%20%3C%3D%2026631500%29' +
    '%20%26%20%28total_debt_to_equity_quarter%20%3E%3D%200%29%20%26%20%28total_debt_to_equity_quarter%20%3C%3D%2026631500%29' +
    '%20%26%20%28interest_coverage_year%20%3E%3D%20-38121800%29%20%26%20%28interest_coverage_year%20%3C%3D%209388700%29' +
    '%20%26%20%28return_on_investment_trailing_12months%20%3E%3D%20-5666000%29%20%26%20%28return_on_investment_trailing_12months%20%3C%3D%2062100%29',
  'agg6': 'return_on_investment_5years%20%3E%3D%20-343400%29%20%26%20%28return_on_investment_5years%20%3C%3D%2019800%29' +
    '%20%26%20%28return_on_investment_year%20%3E%3D%20-5666000%29%20%26%20%28return_on_investment_year%20%3C%3D%2062100%29' +
    '%20%26%20%28return_on_assets_trailing_12months%20%3E%3D%20-316000%29%20%26%20%28return_on_assets_trailing_12months%20%3C%3D%2013107600%29' +
    '%20%26%20%28return_on_assets_5years%20%3E%3D%20-95800%29%20%26%20%28return_on_assets_5years%20%3C%3D%2064400%29' +
    '%20%26%20%28return_on_assets_year%20%3E%3D%20-940500%29%20%26%20%28return_on_assets_year%20%3C%3D%2013107600%29',
  'agg7': 'return_on_equity_trailing_12months%20%3E%3D%20-5666000%29%20%26%20%28return_on_equity_trailing_12months%20%3C%3D%20172100%29' +
    '%20%26%20%28return_on_equity_5years%20%3E%3D%20-147700%29%20%26%20%28return_on_equity_5years%20%3C%3D%2057200%29' +
    '%20%26%20%28return_on_equity_year%20%3E%3D%20-5666000%29%20%26%20%28return_on_equity_year%20%3C%3D%20172100%29' +
    '%20%26%20%28beta%20%3E%3D%20-894%29%20%26%20%28beta%20%3C%3D%20927%29' +
    '%20%26%20%28shares_floating%20%3E%3D%200%29%20%26%20%28shares_floating%20%3C%3D%202671500%29',
  'agg8': 'gross_margin_trailing_12months%20%3E%3D%20-6216500%29%20%26%20%28gross_margin_trailing_12months%20%3C%3D%2052000%29' +
    '%20%26%20%28ebitd_margin_trailing_12months%20%3E%3D%20-1403000000%29%20%26%20%28ebitd_margin_trailing_12months%20%3C%3D%2043043600%29' +
    '%20%26%20%28operating_margin_trailing_12months%20%3E%3D%20-6061000000%29%20%26%20%28operating_margin_trailing_12months%20%3C%3D%20934000000%29' +
    '%20%26%20%28net_profit_margin_percent_trailing_12months%20%3E%3D%20-5834000000%29%20%26%20%28net_profit_margin_percent_trailing_12months%20%3C%3D%20695000000%29' +
    '%20%26%20%28net_income_growth_rate_5years%20%3E%3D%20-7156%29%20%26%20%28net_income_growth_rate_5years%20%3C%3D%2027300%29',
  'agg9': 'revenue_growth_rate_5years%20%3E%3D%20-9184%29%20%26%20%28revenue_growth_rate_5years%20%3C%3D%20141800%29' +
    '%20%26%20%28revenue_growth_rate_10years%20%3E%3D%20-6768%29%20%26%20%28revenue_growth_rate_10years%20%3C%3D%2024900%29' +
    '%20%26%20%28eps_growth_rate_5years%20%3E%3D%20-7502%29%20%26%20%28eps_growth_rate_5years%20%3C%3D%2027200%29' +
    '%20%26%20%28eps_growth_rate_10years%20%3E%3D%20-5000%29%20%26%20%28eps_growth_rate_10years%20%3C%3D%2012600%29' +
    '%20%26%20%28volume%20%3E%3D%200%29%20%26%20%28volume%20%3C%3D%2015212000000%29',
  'agg10': 'average_volume%20%3E%3D%200%29%20%26%20%28average_volume%20%3C%3D%2014683000000%29',
};


/** Retieve live company metrics from Google finance
* @param {Array} existingCompanySymbols (optional) list of existing company
*                    symbols, if a new symbol is loaded it will be added
* @return {Promise} aresolves with the company metrics data
**/
let retrieveCompanies = asyncify(function(existingCompanySymbols) {
  return new Promise(function(resolve, reject) {
    try {
      let companyResults = {};
      let resultFields = ['symbol', 'name'];
      let resultData = [];
      let checkSymbols = false;

      if (existingCompanySymbols && existingCompanySymbols.length) {
        checkSymbols = true;
      }
      Object.keys(fields).forEach((lookupField) => {
        // fetchRequests.push(fetch(baseUrl + fields[lookupField] + suffixUrl));

        let searchResults = awaitify(retrieveMetric(baseUrl +
          fields[lookupField] + suffixUrl));

        searchResults.forEach((result) => {
          /* The first two returns are market cap and price.  Create filtered list
              base don cpmanies with market cap >= 10M and price > 0 */

          let companySymbol = result.ticker;


          // Special handling for market_cap - add items to object
          // if (lookupField === 'market_cap') {
          if (lookupField === 'base') {
            let marketCap = result.columns[0].value;
            let price = result.columns[1].value;

            if (marketCap === '-') {
              marketCap = '';
            } else {
              marketCap = utils.checkForNumber(marketCap);
            }

            if (price === '-') {
              price = '';
            } else {
              price = utils.checkForNumber(price);
            }

            // if price > 0 then add company
            if (price > 0) {
              companyResults[companySymbol] = {};
              companyResults[companySymbol]['name'] = result.title;
              companyResults[companySymbol][result.columns[0].field] = marketCap;
              companyResults[companySymbol][result.columns[1].field] = price;
            }
          } else if (companyResults[companySymbol]) {
            // for all opther fields, only add result if company already exists
            result.columns.forEach((column) => {
              let resultValue = column.value;
              let resultField = column.field;

              // Check for empty values which are listed as '-'
              if (resultValue === '-') {
                resultValue = '';
              } else {
                // Check and format number as required
                resultValue = utils.checkForNumber(resultValue);
              }

              if (resultValue !== '') {
                companyResults[companySymbol][resultField] = resultValue;
              }
            });
          }
        });
      });

      // Work through all values and build data
      Object.keys(companyResults).forEach((company) => {
        let companyData = {};

        /* If check symbols, then compare symbol with list and if the symbol
           is new, call add company */
        if (checkSymbols && existingCompanySymbols.indexOf(company) < 0) {
          // Symbol not found, so add it
          let companyDetails = {
            symbol: company,
            companyName: companyResults[company]['name'],
          };

          dynamoSymbols.addCompany(companyDetails);
        }

        companyData.symbol = company;
        // Loop through company and add each value
        Object.keys(companyResults[company]).forEach((companyVal) => {
          if (resultFields.indexOf(companyVal) === -1) {
            resultFields.push(companyVal);
          }
          companyData[companyVal] = companyResults[company][companyVal];
        });
        resultData.push(companyData);
      });
      // console.log(resultData);
      // utils.writeToCsv(resultData, resultFields, 'company-metrics');

      resolve(resultData);
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
});


/**
 * Returns company metrics which were / are active for a specific date
 *
 * @param {String} url the url to retrieve
 * @return {Object}  Results object:
 *    {
 *      symbol: symbolValue,
 *      ...
 *    }
 */
let retrieveMetric = asyncify(function(url) {
  try {
    let fetchResponse = awaitify(fetch(url));
    let responseBody = awaitify(fetchResponse.text());
    let convertedResponse = responseBody
      .replace(/\\x22/g, '&quot;')
      .replace(/\\x27/g, '&#039;')
      .replace(/\\x26/g, '&amp;')
      .replace(/\\x2F/g, '&#47;')
      .replace(/\\x3E/g, '&gt;')
      .replace(/\\x3C/g, '&lt;');

    let jsonResults = JSON.parse(convertedResponse);

    return jsonResults.searchresults;
  } catch (err) {
    console.error(err);
    return {};
  }
});

/**
 * Returns company metrics which were / are active for a specific date
 *
 * @param {String} symbol the company symbol
 * @param {String} valueDate the date to retrieve (yyyy-mm-dd)
 * @return {Object}  Object in form of:
 *    {
 *      symbol: symbolValue,
 *      ...
 *    }
 */
let returnCompanyMetricValuesForDate = asyncify(function(symbol, valueDate) {
  console.log(`returnCompanyMetricValuesForDate called with symbol: ${symbol}, date: ${valueDate}`);
  return new Promise(function(resolve, reject) {
    if (!valueDate || !utils.isDate(valueDate)) {
      throw new Error('valueDate supplied is invalid: ' + valueDate);
    }

    try {
      let metricValues = {};

      let queryDetails = {
        tableName: 'companyMetrics',
        keyConditionExpression: 'symbol = :symbol and ' +
          'metricsDate <= :metricsDate',
        expressionAttributeValues: {
          ':symbol': symbol,
          ':metricsDate': valueDate,
        },
        reverseOrder: true,
        limit: 1,
      };

      let result = awaitify(dynamodb.queryTable(queryDetails));

      if (result.length > 0) {
        metricValues = result[0];
      }

      resolve(metricValues);
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
});

/**
 * Inserts an indicator value
 * @param {Object} metricValue value to insert in form of:
 *    {
 *      symbol: symbol,
 *      metricsDate: valueDate,
 *      eps: eps,
 *      ...
 *    }
 */
let insertCompanyMetricsValue = asyncify(function(metricValue) {
  if (!metricValue['symbol'] || !metricValue['metricsDate']) {
    console.log('metricValue parameters missing: ' +
      JSON.stringify(metricValue));
    return;
  }

  // console.log('----- Start write company metrics -----');

  try {
    metricValue['yearMonth'] = metricValue['metricsDate']
      .substring(0, 7)
      .replace('-', '');

    let insertDetails = {
      tableName: 'companyMetrics',
      values: metricValue,
      primaryKey: ['symbol', 'metricsDate'],
    };

    awaitify(dynamodb.insertRecord(insertDetails));
  } catch (err) {
    console.log(err);
    throw err;
  }
});

/**
 * Retrieves and processes each company metric information
 */
let updateCompanyMetrics = asyncify(function(recLimits) {
  try {
    console.log('----- Start company metrics retrieval -----');
    console.log(`Executed with: ${JSON.stringify(recLimits)}`);
    let startTiming = utils.getTiming();
    let metricsData = awaitify(retrieveCompanies());
    let duration = utils.getTiming(startTiming);
    console.log(`Retrieve metrics took ${duration} seconds`);

    let startRec = 0;
    let endRec = metricsData.length - 1;

    if (recLimits && recLimits.startRec) {
      startRec = recLimits.startRec;
    }

    if (recLimits && recLimits.endRec && recLimits.endRec < endRec) {
      endRec = recLimits.endRec;
    }

    if (startRec > endRec) {
      console.log(`Nothing to do: start rec: ${startRec}, total number of results: ${endRec}`);
      return;
    }

    let metricsDate = utils.returnDateAsString(Date.now());

    console.log(`Updating company metrics for ${metricsDate} from ${startRec} to ${endRec}`);

    for (let c = startRec; c <= endRec; c++) {
      let companyMetricsRecord = metricsData[c];
      companyMetricsRecord['metricsDate'] = metricsDate;

      // Remove nulls, empty values and -
      // Check through for values with null and remove from object
      // Check for numbers returned as strings and convert
      Object.keys(companyMetricsRecord).forEach((field) => {
        let holdingVal = companyMetricsRecord[field];
        if (holdingVal === null || holdingVal === '' ||
          holdingVal === '-') {
          delete companyMetricsRecord[field];
        } else if (typeof (holdingVal) === 'string' &&
          !isNaN(holdingVal.replace(',', ''))) {
          companyMetricsRecord[field] = Number(holdingVal.replace(',', ''));
        }
      });

      awaitify(insertCompanyMetricsValue(companyMetricsRecord));
    }

    console.log(`Updating company metrics finished`);
  } catch (err) {
    console.log(err);
    throw err;
  }
});

module.exports = {
  updateCompanyMetrics: updateCompanyMetrics,
  returnCompanyMetricValuesForDate: returnCompanyMetricValuesForDate,
  insertCompanyMetricsValue: insertCompanyMetricsValue,
  retrieveCompanies: retrieveCompanies,
};
