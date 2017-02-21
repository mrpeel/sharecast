
const fetch = require('node-fetch');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const dynamoSymbols = require('./dynamo-symbols');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');

let baseUrl = 'https://www.google.com/finance?output=json&start=0&num=5000&noIL=1&q=[currency%20%3D%3D%20%22AUD%22%20%26%20%28exchange%20%3D%3D%20%22ASX%22%29%20%26%20%28';
let suffixUrl = ']&restype=company&ei=X6iZWMmFIMGW0AThhLPoCw';

let fields = {
  'earnings_per_share': 'earnings_per_share%20%3E%3D%20-8121%29%20%26%20%28earnings_per_share%20%3C%3D%203679%29',
  'last_price': 'last_price%20%3E%3D%200%29%20%26%20%28last_price%20%3C%3D%2030400%29',
  'average_200day_price': 'average_200day_price%20%3E%3D%20-1000%29%20%26%20%28average_200day_price%20%3C%3D%20199000%29',
  'price_change_52week': 'price_change_52week%20%3E%3D%20-9947%29%20%26%20%28price_change_52week%20%3C%3D%201420100%29',
  'average_200day_price': 'average_200day_price%20%3E%3D%200%29%20%26%20%28average_200day_price%20%3C%3D%2019900%29',
  'price_to_book': 'price_to_book%20%3E%3D%200%29%20%26%20%28price_to_book%20%3C%3D%20266400%29',
  'market_cap': 'market_cap%20%3E%3D%200%29%20%26%20%28market_cap%20%3C%3D%2014039000000000%29',
  'pe_ratio': 'pe_ratio%20%3E%3D%200%29%20%26%20%28pe_ratio%20%3C%3D%201630100%29',
  'dividend_recent_quarter%': 'dividend_recent_quarter%20%3E%3D%200%29%20%26%20%28dividend_recent_quarter%20%3C%3D%20913%29',
  'dividend_next_quarter': 'dividend_next_quarter%20%3E%3D%200%29%20%26%20%28dividend_next_quarter%20%3C%3D%20223%29',
  'dividend_per_share': 'dividend_per_share%20%3E%3D%200%29%20%26%20%28dividend_per_share%20%3C%3D%20913%29',
  'dividend_next_year': 'dividend_next_year%20%3E%3D%200%29%20%26%20%28dividend_next_year%20%3C%3D%20598%29',
  'dividend_per_share_trailing_12months': 'dividend_per_share_trailing_12months%20%3E%3D%200%29%20%26%20%28dividend_per_share_trailing_12months%20%3C%3D%20913%29',
  'dividend_yield': 'dividend_yield%20%3E%3D%200%29%20%26%20%28dividend_yield%20%3C%3D%2075100%29',
  'dividend_recent_year': 'dividend_recent_year%20%3E%3D%200%29%20%26%20%28dividend_recent_year%20%3C%3D%20582800%29',
  'book_value_per_share_year': 'book_value_per_share_year%20%3E%3D%20-2298%29%20%26%20%28book_value_per_share_year%20%3C%3D%2010100%29',
  'cash_per_share_year': 'cash_per_share_year%20%3E%3D%200%29%20%26%20%28cash_per_share_year%20%3C%3D%206587%29',
  'current_assets_to_liabilities_ratio_year': 'current_assets_to_liabilities_ratio_year%20%3E%3D%200%29%20%26%20%28current_assets_to_liabilities_ratio_year%20%3C%3D%206923200%29',
  'longterm_debt_to_assets_year': 'longterm_debt_to_assets_year%20%3E%3D%200%29%20%26%20%28longterm_debt_to_assets_year%20%3C%3D%20102000000%29',
  'longterm_debt_to_assets_quarter': 'longterm_debt_to_assets_quarter%20%3E%3D%200%29%20%26%20%28longterm_debt_to_assets_quarter%20%3C%3D%20102000000%29',
  'total_debt_to_assets_year': 'total_debt_to_assets_year%20%3E%3D%200%29%20%26%20%28total_debt_to_assets_year%20%3C%3D%20208999999.99999998%29',
  'total_debt_to_assets_quarter': 'total_debt_to_assets_quarter%20%3E%3D%200%29%20%26%20%28total_debt_to_assets_quarter%20%3C%3D%20208999999.99999998%29',
  'longterm_debt_to_equity_year': 'longterm_debt_to_equity_year%20%3E%3D%200%29%20%26%20%28longterm_debt_to_equity_year%20%3C%3D%202892500%29',
  'longterm_debt_to_equity_quarter': 'longterm_debt_to_equity_quarter%20%3E%3D%200%29%20%26%20%28longterm_debt_to_equity_quarter%20%3C%3D%202892500%29',
  'total_debt_to_equity_year': 'total_debt_to_equity_year%20%3E%3D%200%29%20%26%20%28total_debt_to_equity_year%20%3C%3D%2026631500%29',
  'total_debt_to_equity_quarter': 'total_debt_to_equity_quarter%20%3E%3D%200%29%20%26%20%28total_debt_to_equity_quarter%20%3C%3D%2026631500%29',
  'interest_coverage_year': 'interest_coverage_year%20%3E%3D%20-38121800%29%20%26%20%28interest_coverage_year%20%3C%3D%209388700%29',
  'return_on_investment_trailing_12months': 'return_on_investment_trailing_12months%20%3E%3D%20-5666000%29%20%26%20%28return_on_investment_trailing_12months%20%3C%3D%2062100%29',
  'return_on_investment_5years': 'return_on_investment_5years%20%3E%3D%20-343400%29%20%26%20%28return_on_investment_5years%20%3C%3D%2019800%29',
  'return_on_investment_year': 'return_on_investment_year%20%3E%3D%20-5666000%29%20%26%20%28return_on_investment_year%20%3C%3D%2062100%29',
  'return_on_assets_trailing_12months': 'return_on_assets_trailing_12months%20%3E%3D%20-316000%29%20%26%20%28return_on_assets_trailing_12months%20%3C%3D%2013107600%29',
  'return_on_assets_5years': 'return_on_assets_5years%20%3E%3D%20-95800%29%20%26%20%28return_on_assets_5years%20%3C%3D%2064400%29',
  'return_on_assets_year': 'return_on_assets_year%20%3E%3D%20-940500%29%20%26%20%28return_on_assets_year%20%3C%3D%2013107600%29',
  'return_on_equity_trailing_12months': 'return_on_equity_trailing_12months%20%3E%3D%20-5666000%29%20%26%20%28return_on_equity_trailing_12months%20%3C%3D%20172100%29',
  'return_on_equity_5years': 'return_on_equity_5years%20%3E%3D%20-147700%29%20%26%20%28return_on_equity_5years%20%3C%3D%2057200%29',
  'return_on_equity_year': 'return_on_equity_year%20%3E%3D%20-5666000%29%20%26%20%28return_on_equity_year%20%3C%3D%20172100%29',
  'beta': 'beta%20%3E%3D%20-894%29%20%26%20%28beta%20%3C%3D%20927%29',
  'shares_floating': 'shares_floating%20%3E%3D%200%29%20%26%20%28shares_floating%20%3C%3D%202671500%29',
  // 'percent_institutional_held': 'percent_institutional_held%20%3E%3D%200%29%20%26%20%28percent_institutional_held%20%3C%3D%20100%29',
  'gross_margin_trailing_12months': 'gross_margin_trailing_12months%20%3E%3D%20-6216500%29%20%26%20%28gross_margin_trailing_12months%20%3C%3D%2052000%29',
  'ebitd_margin_trailing_12months': 'ebitd_margin_trailing_12months%20%3E%3D%20-1403000000%29%20%26%20%28ebitd_margin_trailing_12months%20%3C%3D%2043043600%29',
  'operating_margin_trailing_12months': 'operating_margin_trailing_12months%20%3E%3D%20-6061000000%29%20%26%20%28operating_margin_trailing_12months%20%3C%3D%20934000000%29',
  'net_profit_margin_percent_trailing_12months': 'net_profit_margin_percent_trailing_12months%20%3E%3D%20-5834000000%29%20%26%20%28net_profit_margin_percent_trailing_12months%20%3C%3D%20695000000%29',
  'net_income_growth_rate_5years': 'net_income_growth_rate_5years%20%3E%3D%20-7156%29%20%26%20%28net_income_growth_rate_5years%20%3C%3D%2027300%29',
  'revenue_growth_rate_5years': 'revenue_growth_rate_5years%20%3E%3D%20-9184%29%20%26%20%28revenue_growth_rate_5years%20%3C%3D%20141800%29',
  'revenue_growth_rate_10years': 'revenue_growth_rate_10years%20%3E%3D%20-6768%29%20%26%20%28revenue_growth_rate_10years%20%3C%3D%2024900%29',
  'eps_growth_rate_5years': 'eps_growth_rate_5years%20%3E%3D%20-7502%29%20%26%20%28eps_growth_rate_5years%20%3C%3D%2027200%29',
  'eps_growth_rate_10years': 'eps_growth_rate_10years%20%3E%3D%20-5000%29%20%26%20%28eps_growth_rate_10years%20%3C%3D%2012600%29',
  'volume': 'volume%20%3E%3D%200%29%20%26%20%28volume%20%3C%3D%2015212000000%29',
  'average_volume': 'average_volume%20%3E%3D%200%29%20%26%20%28average_volume%20%3C%3D%2014683000000%29',
};


let companyResults = {};
/** Retieve live company metrics from Google finance
* @param {Array} existingCompanySymbols (optional) list of existing company
*                    symbols, if a new symbol is loaded it will be added
* @return {Promise} aresolves with the company metrics data
**/
let retrieveCompanies = asyncify(function(existingCompanySymbols) {
  return new Promise(function(resolve, reject) {
    let resultFields = ['symbol', 'name'];
    let resultData = [];
    let checkSymbols = false;

    if (existingCompanySymbols && existingCompanySymbols.length) {
      checkSymbols = true;
    }

    Object.keys(fields).forEach((lookupField) => {
      // fetchRequests.push(fetch(baseUrl + fields[lookupField] + suffixUrl));
      awaitify(fetch(baseUrl + fields[lookupField] + suffixUrl)
        .then((response) => {
          return response.text();
        })
        .then((responseBody) => {
          let convertedResponse = responseBody
            .replace(/\\x22/g, '&quot;')
            .replace(/\\x27/g, '&#039;')
            .replace(/\\x26/g, '&amp;')
            .replace(/\\x2F/g, '&#47;')
            .replace(/\\x3E/g, '&gt;')
            .replace(/\\x3C/g, '&lt;');

          let jsonResults = JSON.parse(convertedResponse);

          jsonResults.searchresults.forEach((result) => {
            if (!companyResults[result.ticker]) {
              companyResults[result.ticker] = {};
            }
            let companyVals = companyResults[result.ticker];
            companyVals.name = result.title;

            // Check for empty values which are listed as '-'
            if (result.columns[0].value === '-') {
              companyVals[result.columns[0].field] = '';
            } else {
              companyVals[result.columns[0].field] = utils.checkForNumber(
                result.columns[0].value);
            }
          });
        })
        .catch((err) => {
          cosole.log(err);
          reject(err);
        })
      );
    });

    // Work through all values and build data
    Object.keys(companyResults).forEach((company) => {
      let companyData = {};

      /* If checl symbols, then compare symbol with list and if the symbol
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
  });
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

  console.log('----- Start write company metrics -----');

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
  }
});

/**
 * Retrieves and processes each company metric information
 */
let updateCompanyMetrics = asyncify(function() {
  try {
    console.log('----- Start company metrics retrieval -----');
    let metricsData = awaitify(retrieveCompanies());

    // console.log(metricsData);

    metricsData.forEach((companyMetricsRecord) => {
      companyMetricsRecord['metricsDate'] = utils.returnDateAsString(
        Date.now());

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
      awaitify(utils.sleep(50));
    });
  } catch (err) {
    console.log(err);
  }
});

// updateCompanyMetrics();

module.exports = {
  updateCompanyMetrics: updateCompanyMetrics,
  returnCompanyMetricValuesForDate: returnCompanyMetricValuesForDate,
  insertCompanyMetricsValue: insertCompanyMetricsValue,
};
