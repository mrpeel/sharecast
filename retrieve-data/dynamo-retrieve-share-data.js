const yahooFinance = require('yahoo-finance');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const symbols = require('./dynamo-symbols');
const finIndicators = require('./dynamo-retrieve-financial-indicator-data');
const metrics = require('./dynamo-retrieve-google-company-data');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const program = require('commander');

const fields = {
  p: 'previous-close',
  y: 'dividend-yield',
  d: 'dividend-per-share',
  q: 'ex-dividend-date',
  c1: 'change',
  p2: 'change-in-percent',
  d1: 'last-trade-date',
  g: 'day-low',
  h: 'day-high',
  l1: 'last-trade-price-only',
  k: '52-week-high',
  j: '52-week-low',
  j1: 'market-capitalization',
  f6: 'float-shares',
  s1: 'shares-owned',
  x: 'stock-exchange',
  j2: 'shares-outstanding',
  v: 'volume',
  e: 'earnings-per-share',
  e7: 'EPS-estimate-current-year',
  b4: 'book-value',
  j4: 'EBITDA',
  p5: 'price-per-sales',
  p6: 'price-per-book',
  r: 'PE-ratio',
  r5: 'PEG-ratio',
  r6: 'price-per-EPS-estimate-current-year',
  r7: 'price-per-EPS-estimate-next-year',
  s7: 'short-ratio',
  s6: 'revenue',
};

const indiceFields = {
  d1: 'last-trade-date',
  p: 'previous-close',
  c1: 'change',
  p2: 'change-in-percent',
  g: 'day-low',
  h: 'day-high',
  k: '52-week-high',
  j: '52-week-low',
  j5: 'change-from-52-week-low',
  k4: 'change-from-52-week-high',
  j6: 'percent-change-from-52-week-low',
  k5: 'percent-change-from-52-week-high',
  n: 'name',
};

let symbolLookup = {};
let indexLookup = {};
let indexSymbols = [];
let companyLookup = {};
let indices = [];
let companies = [];

/* Retrieve index values */
let lastResultDate;
const indiceFieldsToRetrieve = utils.createFieldArray(indiceFields);
const companyFieldsToRetrieve = utils.createFieldArray(fields);
let symbolGroups = [];
let shareRetrievals = [];
let resultFields = [];
let resultData = [];
let indexData = [];
let maxResultDate = '';

/**  Retrieve company and index symbols and sets them up for retrieval
* @return {Object} coninating values in the format:
*  {
*    indexLookup: {Object}, (object to match index symbol with lookup
*                          symbol, e.g. ^ALLORD: ALLORD)
*    indices: [], (array of symbols to use during retrieval)
*    companyLookup: {Object},  (object to match company symbol with lookup
*                          symbol ^AAD: AAD)
*    companies: [], (array of symbols to use during retrieval)
*    symbolLookup: {Object}, (object to match lookup with real index / company
*                          symbol)
*    indexSymbols: [], (array of the index symbols used)
*  }
*/

let setupSymbols = asyncify(function() {
  try {
    console.log('----- Start setup symbols -----');
    let indexValues = awaitify(symbols.getIndices());
    let companyValues = awaitify(symbols.getCompanies());
    let wSymbolLookup = {};
    let wIndexLookup = {};
    let wIndexSymbols = [];
    let wCompanyLookup = {};
    let wIndices = [];
    let wCompanies = [];


    indexValues.forEach((indexValue) => {
      console.log(indexValue);
      wIndexLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
      wSymbolLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
      wIndexSymbols.push(indexValue['symbol']);
    });

    wIndices = utils.createFieldArray(wIndexLookup);

    companyValues.forEach((companyValue) => {
      wCompanyLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
      wSymbolLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
    });

    wCompanies = utils.createFieldArray(wCompanyLookup);

    return {
      indexLookup: wIndexLookup,
      indices: wIndices,
      companyLookup: wCompanyLookup,
      companies: wCompanies,
      symbolLookup: wSymbolLookup,
      indexSymbols: wIndexSymbols,
    };
  } catch (err) {
    console.log(err);
  }
});

/**  Use the max date for an array of index symbols
*     to determine the last date which data was collected
* @param {Array} indexSymbols - one or more index symbols to look up
* @return {String} with date formatted as 'YYYY-MM-DD'
*/
let getLastRetrievalDate = function(indexSymbols) {
  console.log('----- Start get last retrieval date -----');
  return new Promise(function(resolve, reject) {
    try {
      let queryDetails = {
        tableName: 'indexQuotes',
        keyConditionExpression: 'symbol = :symbol',
        reverseOrder: true,
        limit: 1,
        projectionExpression: 'quoteDate',
      };

      let retrievalDate = '';


      indexSymbols.forEach((indexSymbol) => {
        queryDetails.expressionAttributeValues = {
          ':symbol': indexSymbol,
        };
        let result = awaitify(dynamodb.queryTable(queryDetails));

        // Check if this result is later than what we already have
        if (result.length > 0 && result[0]['quoteDate'] > retrievalDate) {
          retrievalDate = result[0]['quoteDate'];
        }
      });

      resolve(retrievalDate);
    } catch (err) {
      console.log(err);
      reject(err);
    }
  });
};


let retrieveSnapshot = function(symbol, fields) {
  return new Promise(function(resolve, reject) {
    let snapshotOptions = {
      fields: fields,
    };

    // Check if one or many symbols
    if (Array.isArray(symbol)) {
      snapshotOptions.symbols = symbol;
    } else {
      snapshotOptions.symbol = symbol;
    }
    yahooFinance.snapshot(snapshotOptions).then((result) => {
      resolve(result);
    }).catch((err) => {
      reject(err);
    });
  });
};

let processResults = function(results) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      // Multiple  symbols returned
      results.forEach((indResult) => {
        processResult(indResult);
      });
    } else {
      // Single symbol returned
      processResult(results);
    }
  }
};

let processResult = function(result) {
  // Retrieve last trade date to check whether to output this value
  result.lastTradeDate = utils.returnDateAsString(result.lastTradeDate);
  if (result.lastTradeDate > lastResultDate) {
    if (result.lastTradeDate > maxResultDate) {
      maxResultDate = result.lastTradeDate;
    }

    // Convert yahoo symbol to generic symbol
    result.symbol = symbolLookup[result.symbol];

    Object.keys(result).forEach((field) => {
      // Check the field is in the csv list
      if (resultFields.indexOf(field) === -1) {
        resultFields.push(field);
      }

      // Reset number here required
      result[field] = utils.checkForNumber(result[field]);
    });
    // Add result to csv data
    resultData.push(result);
  }
};

let writeIndexResults = asyncify(function(indexData) {
  console.log('----- Start write index quote results -----');
  try {
    // Set up the basic insert structure for dynamo
    let insertDetails = {
      tableName: 'indexQuotes',
      values: {},
      primaryKey: [
        'symbol', 'quoteDate',
      ],
    };

    for (let c = 0; c < indexData.length; c++) {
      // Prepare and insert row
      let indexRow = indexData[c];
      indexRow.quoteDate = utils.returnDateAsString(indexRow['lastTradeDate']);
      indexRow.yearMonth = indexRow.quoteDate.substring(0, 7).replace('-', '');

      // Check through for values with null and remove from object
      Object.keys(indexRow).forEach((field) => {
        if (indexRow[field] === null) {
          delete indexRow[field];
        }
      });

      insertDetails.values = indexRow;
      awaitify(dynamodb.insertRecord(insertDetails));
    }
  } catch (err) {
    console.log(err);
  }
});

/**
 * Converts individual index data records into an array of values which
 * can be appended to every company symbol record
 * @param {Array} indexData the data in the format:
 *    {
 *    symbol: ALLORD
 *    lastTradeDate: 2017-02-03
 *     ...
 *    }
 * @return {Array}  Array in the form of
 *    {
 *      "allordpreviousclose": 5696.4,
 *      "allordchange": -23.9,
 *      "allorddayslow": 5668.9,
 *      ...
 *    }
 */
let convertIndexDatatoAppendData = function(indexData) {
  let returnVal = {};

  indexData.forEach((indexRow) => {
    console.log(indexRow);
    indexPrefix = indexRow['symbol'].toLowerCase();
    returnVal[indexPrefix + 'previousclose'] = indexRow['previousClose'];
    returnVal[indexPrefix + 'change'] = indexRow['change'];
    returnVal[indexPrefix + 'dayslow'] = indexRow['daysLow'];
    returnVal[indexPrefix + 'dayshigh'] = indexRow['daysHigh'];
  });

  console.log(returnVal);
  return returnVal;
};

/**
 * Copies properties from two objects to a new object
 * @param {Object} object1
 * @param {Object} object2
 * @return {Object}  a new object with all the properties
 */
let addObjectProperties = function(object1, object2) {
  let returnObj = {};

  Object.keys(object1).forEach((key) => {
    returnObj[key] = object1[key];
  });

  Object.keys(object2).forEach((key) => {
    returnObj[key] = object2[key];
  });

  console.log('----- Appended object -----');
  console.log(JSON.stringify(returnObj));

  return returnObj;
};

/**
 * Appends data to every compamy row
 * @param {Object} dataVals the data in the format:
 *    {
 *    data: base data array
 *    fields: fields list array
 *    append: object with key/value pairs to append to every row
 *    }
 * @return {Object}  Object in the form of
 *    {
 *    data: updated data array
 *    fields: updated fields list array
 *    }
 */
let addAppendDataToRows = function(dataVals) {
  let workingData = dataVals;
  // Check if data is present to append
  if (!workingData.append) {
    return workingData;
  }

  // Check if fields have been supplied
  if (workingData.fields) {
    // Work through array and retrieve the key of each value
    Object.keys(workingData.append).forEach((appendKey) => {
      if (workingData.fields.indexOf(appendKey) < 0) {
        workingData.fields.push(appendKey);
      }
    });
  }

  // Append data to every row
  for (c = 0; c < workingData.data.length; c++) {
    Object.keys(workingData.append).forEach((appendKey) => {
      workingData.data[c][appendKey] = workingData.append[appendKey];
    });
  }

  return workingData;
};

/**
 * Looks up current company Metrics and appends them to each company row
 * @param {Object} dataVals the data in the format:
 *    {
 *    data: base data array
 *    fields: fields list array
 *    }
 * @return {Object}  Object in the form of
 *    {
 *    data: updated data array
 *    fields: updated fields list array
 *    }
 */
let addMetricDataToRows = asyncify(function(dataVals) {
  return new Promise(function(resolve, reject) {
    try {
      let wkData = dataVals;

      // Check if fields have been supplied
      if (wkData.fields) {
        let metricFields = ['EPS', 'PriceToBook', 'MarketCap', 'PE',
          'DividendRecentQuarter', 'DividendNextQuarter', 'DPSRecentYear', 'IAD',
          'DividendPerShare', 'DividendYield', 'Dividend', 'BookValuePerShareYear',
          'CashPerShareYear', 'CurrentRatioYear', 'LTDebtToAssetsYear',
          'LTDebtToAssetsQuarter', 'TotalDebtToAssetsYear',
          'TotalDebtToAssetsQuarter', 'LTDebtToEquityYear', 'LTDebtToEquityQuarter',
          'TotalDebtToEquityYear', 'TotalDebtToEquityQuarter', 'AINTCOV',
          'ReturnOnInvestmentTTM', 'ReturnOnInvestment5Years',
          'ReturnOnInvestmentYear', 'ReturnOnAssetsTTM', 'ReturnOnAssets5Years',
          'ReturnOnAssetsYear', 'ReturnOnEquityTTM', 'ReturnOnEquity5Years',
          'ReturnOnEquityYear', 'Beta', 'Float', 'GrossMargin', 'EBITDMargin',
          'OperatingMargin', 'NetProfitMarginPercent', 'NetIncomeGrowthRate5Years',
          'RevenueGrowthRate5Years', 'RevenueGrowthRate10Years',
          'EPSGrowthRate5Years', 'EPSGrowthRate10Years'];

        // Make sure the metrics fields are added to the fields list
        metricFields.forEach((field) => {
          if (wkData.fields.indexOf(field) === -1) {
            wkData.fields.push(field);
          }
        });
      }

      // Append data to every row
      for (let c = 0; c < wkData.data.length; c++) {
        let companSymbol = wkData.data[c]['symbol'];

        awaitify(
          metrics.returnCompanyMetricValuesForDate(
            companSymbol, utils.returnDateAsString(Date.now()))
            .then((meVal) => {
              if (meVal['CompanySymbol']) {
                wkData.data[c]['EPS'] = meVal['EPS'];
                wkData.data[c]['PriceToBook'] = meVal['PriceToBook'];
                wkData.data[c]['MarketCap'] = meVal['MarketCap'];
                wkData.data[c]['PE'] = meVal['PE'];
                wkData.data[c]['DividendRecentQuarter'] = meVal['DividendRecentQuarter'];
                wkData.data[c]['DividendNextQuarter'] = meVal['DividendNextQuarter'];
                wkData.data[c]['DPSRecentYear'] = meVal['DPSRecentYear'];
                wkData.data[c]['IAD'] = meVal['IAD'];
                wkData.data[c]['DividendPerShare'] = meVal['DividendPerShare'];
                wkData.data[c]['DividendYield'] = meVal['DividendYield'];
                wkData.data[c]['Dividend'] = meVal['Dividend'];
                wkData.data[c]['BookValuePerShareYear'] = meVal['BookValuePerShareYear'];
                wkData.data[c]['CashPerShareYear'] = meVal['CashPerShareYear'];
                wkData.data[c]['CurrentRatioYear'] = meVal['CurrentRatioYear'];
                wkData.data[c]['LTDebtToAssetsYear'] = meVal['LTDebtToAssetsYear'];
                wkData.data[c]['LTDebtToAssetsQuarter'] = meVal['LTDebtToAssetsQuarter'];
                wkData.data[c]['TotalDebtToAssetsYear'] = meVal['TotalDebtToAssetsYear'];
                wkData.data[c]['TotalDebtToAssetsQuarter'] = meVal['TotalDebtToAssetsQuarter'];
                wkData.data[c]['LTDebtToEquityYear'] = meVal['LTDebtToEquityYear'];
                wkData.data[c]['LTDebtToEquityQuarter'] = meVal['LTDebtToEquityQuarter'];
                wkData.data[c]['TotalDebtToEquityYear'] = meVal['TotalDebtToEquityYear'];
                wkData.data[c]['TotalDebtToEquityQuarter'] = meVal['TotalDebtToEquityQuarter'];
                wkData.data[c]['AINTCOV'] = meVal['AINTCOV'];
                wkData.data[c]['ReturnOnInvestmentTTM'] = meVal['ReturnOnInvestmentTTM'];
                wkData.data[c]['ReturnOnInvestment5Years'] = meVal['ReturnOnInvestment5Years'];
                wkData.data[c]['ReturnOnInvestmentYear'] = meVal['ReturnOnInvestmentYear'];
                wkData.data[c]['ReturnOnAssetsTTM'] = meVal['ReturnOnAssetsTTM'];
                wkData.data[c]['ReturnOnAssets5Years'] = meVal['ReturnOnAssets5Years'];
                wkData.data[c]['ReturnOnAssetsYear'] = meVal['ReturnOnAssetsYear'];
                wkData.data[c]['ReturnOnEquityTTM'] = meVal['ReturnOnEquityTTM'];
                wkData.data[c]['ReturnOnEquity5Years'] = meVal['ReturnOnEquity5Years'];
                wkData.data[c]['ReturnOnEquityYear'] = meVal['ReturnOnEquityYear'];
                wkData.data[c]['Beta'] = meVal['Beta'];
                wkData.data[c]['Float'] = meVal['Float'];
                wkData.data[c]['GrossMargin'] = meVal['GrossMargin'];
                wkData.data[c]['EBITDMargin'] = meVal['EBITDMargin'];
                wkData.data[c]['OperatingMargin'] = meVal['OperatingMargin'];
                wkData.data[c]['NetProfitMarginPercent'] = meVal['NetProfitMarginPercent'];
                wkData.data[c]['NetIncomeGrowthRate5Years'] = meVal['NetIncomeGrowthRate5Years'];
                wkData.data[c]['RevenueGrowthRate5Years'] = meVal['RevenueGrowthRate5Years'];
                wkData.data[c]['RevenueGrowthRate10Years'] = meVal['RevenueGrowthRate10Years'];
                wkData.data[c]['EPSGrowthRate5Years'] = meVal['EPSGrowthRate5Years'];
                wkData.data[c]['EPSGrowthRate10Years'] = meVal['EPSGrowthRate10Years'];
              }
              console.log(wkData.data[c]);
            })
            .catch((err) => {
              console.log(err);
            })
        );
      }

      resolve(wkData);
    } catch (err) {
      reject(err);
    }
  });
});

let writeCompanyQuoteData = asyncify(function(quoteObject, quoteDate) {
  let insertDetails = {
    tableName: 'companyQuotes',
    values: {},
    primaryKey: [
      'symbol', 'quoteDate',
    ],
  };

  // Iterate through quotes and save each one
  quoteObject.forEach((quote) => {
    quote.quoteDate = quoteDate;
    quote.yearMonth = quoteDate.substring(0, 7).replace('-', '');

    // Check through for values with null and remove from object
    Object.keys(quote).forEach((field) => {
      if (quote[field] === null) {
        delete quote[field];
      }
    });

    insertDetails.values = quote;
    awaitify(dynamodb.insertRecord(insertDetails));
  });
});

let executeFinancialIndicators = asyncify(function() {
  awaitify(finIndicators.updateIndicatorValues());
});

let executeCompanyMetrics = asyncify(function() {
  let symbolResult = awaitify(setupSymbols());
  let mCompanies = symbolResult.companies;

  awaitify(metrics.updateCompanyMetrics(mCompanies));
});


let executeQuoteRetrieval = asyncify(function() {
  let dataToAppend = {};
  let indexDataToAppend = {};
  let symbolResult = awaitify(setupSymbols());

  symbolLookup = symbolResult.symbolLookup;
  indexLookup = symbolResult.indexLookup;
  indexSymbols = symbolResult.indexSymbols;
  companyLookup = symbolResult.companyLookup;
  indices = symbolResult.indices;
  companies = symbolResult.companies;

  lastResultDate = awaitify(getLastRetrievalDate(indexSymbols));

  let todayString = utils.returnDateAsString(Date.now());
  let financialIndicatos = awaitify(finIndicators
    .returnIndicatorValuesForDate(todayString));

  console.log('----- Start retrieve index quotes -----');

  retrieveSnapshot(indices, indiceFieldsToRetrieve)
    .then((results) => {
      return processResults(results);
    })
    .then(function() {
      // Write data to index tables
      indexData = resultData;

      if (indexData.length > 0) {
        indexDataToAppend = convertIndexDatatoAppendData(indexData);
        writeIndexResults(indexData);
      } else {
        console.log('No new index data to save');
      }

      return true;
    }).then(function() {
    // Reset fields for companies
    resultFields = [];
    resultData = [];

    console.log('----- Start retrieve company quotes -----');

    // Split companies into groups of 10 so each request contains 10
    for (companyCounter = 0; companyCounter < companies.length;
      companyCounter += 10) {
      symbolGroups.push(companies.slice(companyCounter, companyCounter + 10));
    }

    symbolGroups.forEach((symbolGroup) => {
      shareRetrievals.push(retrieveSnapshot(symbolGroup,
        companyFieldsToRetrieve));
    });

    // When all returns are back, process the results
    Promise.all(shareRetrievals).then((results) => {
      results.forEach((result) => {
        processResults(result);
      });

      if (resultData.length > 0) {
        // create append array from indicators and index data
        dataToAppend = addObjectProperties(financialIndicatos,
          indexDataToAppend);

        let updatedResults = addAppendDataToRows({
          data: resultData,
          fields: resultFields,
          append: dataToAppend,
        });

        addMetricDataToRows(updatedResults)
          .then((resultsWithMetrics) => {
            // console.log(resultsWithMetrics.fields);

            console.log('----- Start write company quotes -----');

            writeCompanyQuoteData(resultsWithMetrics.data, maxResultDate);

            utils.writeToCsv(resultsWithMetrics.data, resultsWithMetrics.fields,
              'companies', maxResultDate);
          });
      } else {
        console.log('No new company data to save');
      }
    }).catch((err) => {
      console.log(err);
    });
  })
    .catch((err) => {
      console.log(err);
    });
});

program
  .version('0.0.1')
  .description('Sharecast share data retrieval')
  .option('-f, --financial', 'Retrieve financial indicators')
  .option('-m, --metrics', 'Retrieve company metrics')
  .option('-q, --quotes', 'Retrieve index and company quotes')
  .parse(process.argv);

if (program.financial) {
  console.log('Executing retrieve financial indicators');
  executeFinancialIndicators();
}
if (program.metrics) {
  console.log('Executing retrieve company metrics');
  executeCompanyMetrics();
}
if (program.quotes) {
  console.log('Executing retrieve index and company quotes');
  executeQuoteRetrieval();
}
