'use strict';

const yahooFinance = require('yahoo-finance');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
const symbols = require('./dynamo-symbols');
const finIndicators = require('./dynamo-retrieve-financial-indicator-data');
const metrics = require('./dynamo-retrieve-google-company-data');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
// const program = require('commander');

/* const fields = {
  p: 'previous-close',
  y: 'dividend-yield',
  d: 'dividend-per-share',
  q: 'ex-dividend-date',
  r1: 'dividend-pay-date',
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

/* Retrieve index values */
/* const indiceFieldsToRetrieve = utils.createFieldArray(indiceFields);
const companyFieldsToRetrieve = utils.createFieldArray(fields); */

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
let setupSymbols = asyncify(function(indicesOnly) {
  try {
    console.log('----- Start setup symbols -----');
    let indexValues = awaitify(symbols.getIndices());
    let wSymbolLookup = {};
    let wIndexLookup = {};
    let wIndexSymbols = [];
    let wCompanyLookup = {};
    let wIndices = [];
    let wCompanies = [];
    let returnVal = {};


    indexValues.forEach((indexValue) => {
      console.log(indexValue);
      wIndexLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
      wSymbolLookup[indexValue['yahoo-symbol']] = indexValue['symbol'];
      wIndexSymbols.push(indexValue['symbol']);
    });

    wIndices = utils.createFieldArray(wIndexLookup);

    returnVal = {
      indexLookup: wIndexLookup,
      indices: wIndices,
      symbolLookup: wSymbolLookup,
      indexSymbols: wIndexSymbols,
    };

    if (!indicesOnly) {
      let companyValues = awaitify(symbols.getCompanies());

      companyValues.forEach((companyValue) => {
        wCompanyLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
        wSymbolLookup[companyValue['yahoo-symbol']] = companyValue['symbol'];
      });

      wCompanies = utils.createFieldArray(wCompanyLookup);

      returnVal.companyLookup = wCompanyLookup;
      returnVal.companies = wCompanies;
    }

    return returnVal;
  } catch (err) {
    console.log(err);
  }
});

/**  Call yahoo quote and map values to known structure
* @param {Array} symbols - a string or array of strings with yahoo symbols to
*                         look up
* @param {String} type - 'index' or 'company' symbol
* @return {Array} an array of results
*/
let retrieveSnapshot = function(symbols, type) {
  if (!symbols) {
    console.error('retrieveSnapshot error: no symbols supplied');
    return;
  }

  if (!type) {
    console.error('retrieveSnapshot error: no type supplied');
    return;
  }

  return new Promise(function(resolve, reject) {
    let retrieveResults = [];
    let returnResults = [];
    // If a single value, convert to an array
    if (!Array.isArray(symbols)) {
      symbols = [symbols];
    }

    symbols.forEach((index) => {
      retrieveResults.push(retrieveQuote(index));
    });

    Promise.all(retrieveResults).then((results) => {
      results.forEach((result) => {
        returnResults.push(mapFields(result, type));
      });

      resolve(returnResults);
    }).catch((err) => {
      reject(err);
    });
  });
};
/**  Return current quote information from yahoo finance
* @param {Array} symbol - a string or array of strings with yahoo symbols to
*                         look up
* @return {Array} an array of results
*/
let retrieveQuote = function(symbol) { // }, fields) {
  return new Promise(function(resolve, reject) {
    /* let snapshotOptions = {
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
    }); */

    let quoteOptions = {
      modules: ['price', 'summaryDetail', 'defaultKeyStatistics',
        'financialData'],
    };
    quoteOptions.symbol = symbol;

    yahooFinance.quote(quoteOptions).then((result) => {
      resolve(result);
    }).catch((err) => {
      console.log(err);
      resolve([]);
    // reject(err);
    });
  });
};

/**  Map the return from the update yahoo API to the previous quote structure
* @param {Object} quote - the quote object from Yahoo
* @param {String} symbolType - index or company symbol
* @return {Object} quote restructured into previous format
*/
let mapFields = function(quote, symbolType) {
  let updatedQuote;
  let quotePrice;

  // Put in empty default structures where missing
  quote.summaryDetail = quote.summaryDetail || {};
  quote.regularMarketChange = quote.regularMarketChange || {};
  quote.price = quote.price || {};
  quote.defaultKeyStatistics = quote.defaultKeyStatistics || {};
  quote.financialData = quote.financialData || {};

  if (symbolType === 'index') {
    updatedQuote = {
      '52WeekHigh': quote.summaryDetail.fiftyTwoWeekHigh || null,
      '52WeekLow': quote.summaryDetail.fiftyTwoWeekLow || null,
      'change': quote.price.regularMarketChange || null,
      'changeInPercent': quote.price.regularMarketChangePercent || null,
      'daysHigh': quote.summaryDetail.dayHigh || null,
      'daysLow': quote.summaryDetail.dayLow || null,
      'name': quote.price.shortName || null,
      'previousClose': quote.summaryDetail.previousClose || null,
      'symbol': quote.price.symbol || null,
      'lastTradeDate': quote.price.regularMarketTime || null,
    };

    // Set price to use for 52 week comparisons
    quotePrice = quote.summaryDetail.previousClose || null;
  } else {
    updatedQuote = {
      '52WeekHigh': quote.summaryDetail.fiftyTwoWeekHigh || null,
      '52WeekLow': quote.summaryDetail.fiftyTwoWeekLow || null,
      'bookValue': quote.defaultKeyStatistics.bookValue || null,
      'change': quote.price.regularMarketChange || null,
      'changeInPercent': quote.price.regularMarketChangePercent || null,
      'daysHigh': quote.summaryDetail.dayHigh || null,
      'daysLow': quote.summaryDetail.dayLow || null,
      'earningsPerShare': quote.defaultKeyStatistics.forwardEps || null,
      'ebitda': quote.financialData.ebitda || null,
      'exDividendDate': quote.summaryDetail.exDividendDate || null,
      'exDividendPayout': quote.summaryDetail.dividendRate || null,
      'fiveYearAvgDividendYield': quote.summaryDetail.fiveYearAvgDividendYield || null,
      'Float': quote.defaultKeyStatistics.floatShares || null,
      'lastTradePriceOnly': quote.price.regularMarketPrice || null,
      'name': quote.price.shortName || null,
      'pegRatio': quote.defaultKeyStatistics.pegRatio || null,
      'peRatio': quote.summaryDetail.trailingPE || null,
      'previousClose': quote.summaryDetail.previousClose || null,
      'price200DayAverage': quote.summaryDetail.twoHundredDayAverage || null,
      'price52WeekPercChange': quote.defaultKeyStatistics['52WeekChange'] || null,
      'pricePerBook': quote.defaultKeyStatistics.priceToBook || null,
      'symbol': quote.price.symbol || null,
      'lastTradeDate': quote.price.regularMarketTime || null,
      'volume': quote.summaryDetail.volume || null,
    };

    // Set price to use for 52 week comparisons
    quotePrice = quote.price.regularMarketPrice || null;
  }

  // Calculate 52 week changes
  if (quotePrice) {
    if (updatedQuote['52WeekHigh']) {
      updatedQuote['changeFrom52WeekHigh'] = quotePrice - updatedQuote['52WeekHigh'];
      updatedQuote['percebtChangeFrom52WeekHigh'] = (quotePrice -
      updatedQuote['52WeekHigh']) /
      updatedQuote['52WeekHigh'];
    } else {
      updatedQuote['percebtChangeFrom52WeekHigh'] = 0;
      updatedQuote['changeFrom52WeekHigh'] = 0;
    }
    if (updatedQuote['52WeekLow']) {
      updatedQuote['changeFrom52WeekLow'] = quotePrice - updatedQuote['52WeekLow'];
      updatedQuote['percentChangeFrom52WeekLow'] = (quotePrice -
      updatedQuote['52WeekLow']) /
      updatedQuote['52WeekLow'];
    } else {
      updatedQuote['percentChangeFrom52WeekLow'] = 0;
      updatedQuote['changeFrom52WeekLow'] = 0;
    }
  }

  return updatedQuote;
};

/**  Process an array of index quote results
* @param {Array} results - an array with index quote values
*                         look up or  a single index quote value
* @param {Array} symbolLookup - an array of key/value pairs to translate
*                               the symbols yahoo uses to the normal symbol
*/
let processIndexResults = asyncify(function(results, symbolLookup) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      // Multiple  symbols returned
      results.forEach((indResult) => {
        awaitify(processResult(indResult, symbolLookup));
      });
    } else {
      // Single symbol returned
      awaitify(processResult(results, symbolLookup));
    }
  }
});

/**  Process a a single index quote result and insert in dynamo.  Fields are
*     reformatted as reuired, e.g. 1M -> 1000000, and empty fields are removed
* @param {Object} result - the result values
* @param {Array} symbolLookup - an array of key/value pairs to translate
*                               the symbols yahoo uses to the normal symbol
*/
let processResult = asyncify(function(result, symbolLookup) {
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  Object.keys(result).forEach((field) => {
    // Reset number here required
    if (result[field]) {
      result[field] = utils.checkForNumber(result[field]);
    }
  });

  awaitify(writeIndexQuote(result));
});

/**  Process an array of company quote results
* @param {Array} results - an array with index quote values
*                         look up or  a single index quote value
* @param {Array} symbolLookup - an array of key/value pairs to translate
*                               the symbols yahoo uses to the normal symbol
*/
let processCompanyResults = asyncify(function(results, symbolLookup,
  dataToAppend, dividends) {
  if (results) {
    // Check if multi-dimensional
    if (Array.isArray(results)) {
      // Multiple  symbols returned
      results.forEach((indResult) => {
        awaitify(processCompanyResult(indResult, symbolLookup,
          dataToAppend, dividends));
      });
    } else {
      // Single symbol returned
      awaitify(processCompanyResult(results, symbolLookup,
        dataToAppend, dividends));
    }
  }
});

/**  Process a a single company quote result and insert in dynamo.  Fields are
*     reformatted as reuired, e.g. 1M -> 1000000, and empty fields are removed
* @param {Object} result - the result values
* @param {Array} symbolLookup - an array of key/value pairs to translate
*                               the symbols yahoo uses to the normal symbol
*/
let processCompanyResult = asyncify(function(result, symbolLookup,
  dataToAppend, dividends) {
  // Convert yahoo symbol to generic symbol
  result.symbol = symbolLookup[result.symbol];

  Object.keys(result).forEach((field) => {
    // Reset number here required
    if (result[field]) {
      result[field] = utils.checkForNumber(result[field]);
    }
  });

  // Add metric data to this result
  // result = awaitify(addMetricsToQuote(result));

  // If data to append,ietrate through keys and add it to the record
  if (dataToAppend) {
    Object.keys(dataToAppend).forEach((dataKey) => {
      result[dataKey] = dataToAppend[dataKey];
    });
  }

  // Add dividends details if located
  if (dividends && dividends[result.symbol]) {
    result['exDividendDate'] = dividends[result.symbol]['exDividendDate'];
    result['exDividendPayout'] = dividends[result.symbol]['exDividendPayout'];
  }

  // Write value
  awaitify(writeCompanyQuoteData(result));
});

/**  Return an array of dividens results from yahoo finance for a specified
*     time period
* @param {Array} symbol - a string or array of strings with yahoo symbols to
*                         look up
* @param {String} startDate - the start of the timer period
* @param {String} endDate - the end of the timer period
* @return {Array} an array of results
*/
let retrieveDividends = function(symbol, startDate, endDate) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      period: 'v',
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

/**  Write an index quote record to dynamodb.
*    Converts lastTradeDate -> quoteDate and checks and removes anu invalid
*     values are.
* @param {Object} indexQuote - the index quote to write
*/
let writeIndexQuote = asyncify(function(indexQuote) {
  // console.log('----- Write index quote  -----');
  try {
    // Set up the basic insert structure for dynamo
    let insertDetails = {
      tableName: 'indexQuotes',
      values: {},
      primaryKey: [
        'symbol', 'quoteDate',
      ],
    };

    indexQuote.quoteDate = utils
      .returnDateAsString(indexQuote['lastTradeDate']);
    indexQuote.yearMonth = indexQuote.quoteDate
      .substring(0, 7).replace('-', '');

    // Check through for values with null and remove from object
    Object.keys(indexQuote).forEach((field) => {
      if (indexQuote[field] === null || indexQuote[field] === 'Infinity'
        || indexQuote[field] === '-Infinity') {
        delete indexQuote[field];
      }
    });

    insertDetails.values = indexQuote;
    awaitify(dynamodb.insertRecord(insertDetails));
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
    // console.log(indexRow);
    let indexPrefix = indexRow['symbol'].toLowerCase();
    returnVal[indexPrefix + 'previousclose'] = indexRow['previousClose'];
    returnVal[indexPrefix + 'change'] = indexRow['change'];
    returnVal[indexPrefix + 'dayslow'] = indexRow['daysLow'];
    returnVal[indexPrefix + 'dayshigh'] = indexRow['daysHigh'];
    returnVal[indexPrefix + 'percebtChangeFrom52WeekHigh'] = indexRow['percebtChangeFrom52WeekHigh'];
    returnVal[indexPrefix + 'percentChangeFrom52WeekLow'] = indexRow['percentChangeFrom52WeekLow'];
  });

  console.log(returnVal);
  return returnVal;
};

/**
 * Retrieves index values for a specific date stored in dynamodb
 * @param {String} dateVal the date
 * @return {Array}  Array in the form of
 *    {
 *      "allordpreviousclose": 5696.4,
 *      "allordchange": -23.9,
 *      "allorddayslow": 5668.9,
 *      ...
 *    }
 */
let returnIndexDataForDate = function(dateVal) {
  if (!utils.isDate(dateVal)) {
    console.error(`Invalid dateVal: ${dateVal}`);
    return;
  }
  let returnVal = {};
  let quoteDate = utils.returnDateAsString(dateVal);
  let allIndiceResults = [];

  let queryDetails = {
    tableName: 'indexQuotes',
    keyConditionExpression: 'symbol = :symbol and quoteDate <= :quoteDate',
    expressionAttributeValues: {
      ':quoteDate': quoteDate,
    },
    reverseOrder: true,
    limit: 1,
  };


  let symbolResult = awaitify(setupSymbols(true));
  let indexSymbols = symbolResult.indexSymbols;

  indexSymbols.forEach((index) => {
    queryDetails.expressionAttributeValues[':symbol'] = index;
    let indexResults = awaitify(dynamodb.queryTable(queryDetails));
    if (indexResults.length) {
      allIndiceResults = allIndiceResults.concat(indexResults);
    }
  });


  // Check we got a metrics result
  if (allIndiceResults.length) {
    returnVal = convertIndexDatatoAppendData(allIndiceResults);
  }

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

/**  Write a conpany quote record to dynamodb.
*    Converts lastTradeDate -> quoteDate, copies lastTradePriceOnly ->
*     adjustedPrice, and checks and removes any invalid values.
* @param {Object} quoteData - the company quote to write
*/
let writeCompanyQuoteData = asyncify(function(quoteData) {
  // If unexpected r3cords come back with no trade data, skop them
  if (!quoteData['lastTradeDate']) {
    return;
  }

  let insertDetails = {
    tableName: 'companyQuotes',
    values: {},
    primaryKey: [
      'symbol', 'quoteDate',
    ],
  };

  quoteData.quoteDate = utils.returnDateAsString(quoteData['lastTradeDate']);
  quoteData.yearMonth = quoteData['quoteDate'].substring(0, 7).replace('-', '');
  if (!quoteData.adjustedPrice) {
    quoteData.adjustedPrice = quoteData['lastTradePriceOnly'];
  }

  delete quoteData['lastTradeDate'];

  // Check through for values with null and remove from object
  Object.keys(quoteData).forEach((field) => {
    if (quoteData[field] === null || quoteData[field] === ''
      || quoteData[field] === 'Infinity' || quoteData[field] === '-Infinity') {
      delete quoteData[field];
    }
  });

  insertDetails.values = quoteData;
  return awaitify(dynamodb.insertRecord(insertDetails));
});

/**  Processes updates to company quotes for a specific date.  Looks up the
*     relevant company metrics for the date, and processes an update to the
*     company quote record to add the metrics values.
* @param {Array} symbols - all companies which are being processed
* @param {String} quoteDate - the date to process
* @param {Object} recLimits (optional) - specifies where to start and stop
*     the operation, in form of:
*     {
        startRec: 100,
        endRec, 199,
*      }
*     This would process from record index 100 to 199 in the symbols list
*/
let updateQuotesWithMetrics = asyncify(function(symbols, quoteDate, recLimits) {
  if (!symbols) {
    console.log('updateQuotesWithMetrics error: no symbols supplied');
    return;
  }

  if (!quoteDate) {
    console.log('updateQuotesWithMetrics error: no quoteDate supplied');
    return;
  }

  let startRec = 0;
  let endRec = symbols.length - 1;

  if (recLimits && recLimits.startRec) {
    startRec = recLimits.startRec;
  }

  if (recLimits && recLimits.endRec && recLimits.endRec < endRec) {
    endRec = recLimits.endRec;
  }

  if (startRec > endRec) {
    return;
  }

  console.log('Updating company quotes with metrics for ', quoteDate, ' from ',
    startRec, ' to ', endRec);

  let metricsDate = quoteDate;

  // Set-up update details, only update the record if it is found
  let updateDetails = {
    tableName: 'companyQuotes',
    key: {
      quoteDate: metricsDate,
    },
    conditionExpression: 'attribute_exists(symbol) and ' +
      'attribute_exists(quoteDate)',
  };

  for (let c = startRec; c <= endRec; c++) {
    let companySymbol = symbols[c];
    // symbols.forEach((companySymbol) => {
    // If at least one quote record has been returned, continue
    let fieldsPresent = [];
    let updateExpression;
    let expressionAttributeValues = {};
    let expressionAttributeNames = {};

    // Set-up query and retrieve metrics value
    let queryDetails = {
      tableName: 'companyMetrics',
      keyConditionExpression: 'symbol = :symbol and ' +
        'metricsDate <= :metricsDate',
      expressionAttributeValues: {
        ':metricsDate': metricsDate,
        ':symbol': companySymbol,
      },
      reverseOrder: true,
      limit: 1,
    };

    let metricsResult = awaitify(dynamodb.queryTable(queryDetails));

    // Check we got a metrics result
    if (metricsResult.length) {
      let workingRecord = metricsResult[0];

      // Set up the key: symbol and quoteDate
      updateDetails.key.symbol = companySymbol;

      // Remove the attributes which should not be copied into cpmanyQuotes
      delete workingRecord['symbol'];
      delete workingRecord['metricsDate'];
      delete workingRecord['yearMonth'];
      delete workingRecord['created'];

      // Get a list of fields and values to copy
      Object.keys(workingRecord).forEach((field) => {
        expressionAttributeValues[(':' + field)] = workingRecord[field];
        expressionAttributeNames[('#' + field)] = field;
        fieldsPresent.push('#' + field + '=:' + field);
      });

      // Enure that at least one field is present to update
      if (fieldsPresent.length) {
        updateExpression = 'set ' + fieldsPresent.join(',');

        updateDetails.updateExpression = updateExpression;
        updateDetails.expressionAttributeValues = expressionAttributeValues;
        updateDetails.expressionAttributeNames = expressionAttributeNames;

        try {
          awaitify(dynamodb.updateRecord(updateDetails));
        } catch (err) {
          console.log(err);
        }
      }
    }
  }
});

/**  Find the most recent date which quotes were retrieved for.  Because
*     inidividual companies may or may not have a quote for a given date, it
*     uses the last date for the All Ordinaries index
* @return {String} the most recent date found in format YYYY-MM-DD
*/
let getCurrentExecutionDate = asyncify(function() {
  // Get current dat based on last index quote for All Ordinaries
  let queryDetails = {
    tableName: 'indexQuotes',
    keyConditionExpression: 'symbol = :symbol',
    reverseOrder: true,
    expressionAttributeValues: {
      ':symbol': 'ALLORD',
    },
    limit: 1,
    projectionExpression: 'symbol, quoteDate',
  };

  let indexLookup = awaitify(dynamodb.queryTable(queryDetails));
  if (indexLookup.length) {
    return indexLookup[0]['quoteDate'];
  } else {
    return null;
  }
});

/**  Executes the retrieval and storage of the latest financial indicator
*     information
*/
let executeFinancialIndicators = asyncify(function() {
  awaitify(finIndicators.updateIndicatorValues());
});

/**  Executes the retrieval and storage of the latest company metrics
*     information
*/
let executeCompanyMetrics = asyncify(function() {
  let symbolResult = awaitify(setupSymbols());
  let mCompanies = symbolResult.companies;

  awaitify(metrics.updateCompanyMetrics(mCompanies));
});

/**  Executes the retrieval and storage of the latest index quote
*     information
*/
let executeIndexQuoteRetrieval = asyncify(function() {
  let symbolResult = awaitify(setupSymbols(true));

  let symbolLookup = symbolResult.symbolLookup;
  let indices = symbolResult.indices;

  console.log('----- Start retrieve index quotes -----');
  try {
    let results = awaitify(retrieveSnapshot(indices, 'index'));

    awaitify(processIndexResults(results, symbolLookup));
  } catch (err) {
    console.log(err);
  }
});

/**  Executes the retrieval and storage of the latest company quote
*     information
* @param {Object} recLimits   (optional) - specifies where to start and stop
*     the operation, in form of:
*     {
        startRec: 100,
        endRec, 199,
*      }
*     This would process from company record index 100 to 199
*/
let executeCompanyQuoteRetrieval = asyncify(function(recLimits) {
  let dataToAppend = {};
  let symbolGroups = [];
  let symbolResult = awaitify(setupSymbols());

  let symbolLookup = symbolResult.symbolLookup;
  let companies = symbolResult.companies;

  let todayString = utils.returnDateAsString(Date.now());
  let financialIndicatos = awaitify(finIndicators
    .returnIndicatorValuesForDate(todayString));
  let indexDataToAppend = awaitify(returnIndexDataForDate(todayString));

  // create append array from indicators and index data
  dataToAppend = addObjectProperties(financialIndicatos,
    indexDataToAppend);

  let startRec = 0;
  let endRec = companies.length - 1;

  if (recLimits && recLimits.startRec) {
    startRec = recLimits.startRec;
  }

  if (recLimits && recLimits.endRec && recLimits.endRec < endRec) {
    endRec = recLimits.endRec;
  }

  if (startRec > endRec) {
    return;
  }

  console.log('Retrieving company quotes from ', startRec, ' to ', endRec);


  /* Split companies into groups of 15 to ensure request doesn't exceed api
      url length */
  for (let companyCounter = startRec; companyCounter < endRec;
    companyCounter += 15) {
    symbolGroups.push(companies.slice(companyCounter, companyCounter + 15));
  }

  symbolGroups.forEach((symbolGroup) => {
    try {
      let result = awaitify(retrieveSnapshot(symbolGroup,
        'company'));

      // Retrieve dividends
      let dividends = awaitify(getDividendsforDate(symbolGroup,
        todayString, symbolLookup));

      awaitify(processCompanyResults(result, symbolLookup,
        dataToAppend, dividends));
    } catch (err) {
      console.log(err);
    }
  });
});

/**  Executes the retrieval and return of current company dividend
*      information for the date specified
* @param {Array} symbolGroup  one or more yahoo symbols to look up for dividend
* @param {String} retrievalDate - the date to retrieve dividends for
* @param {Array} symbolLookup - an array of key/value pairs to translate
*                               the symbols yahoo uses to the normal symbol
* @return {Object} an object for each company which has relevant dividend
*     information for the specified date in the form of:
*   {
*   'JBH': {
*           'exDividendDate': '2017-02-23',
*           'exDividendPayout': 1.02344,
*           },
*   }
*/
let getDividendsforDate = asyncify(function(symbolGroup, retrievalDate,
  symbolLookup) {
  if (!symbolGroup) {
    console.error('getDividendsforDate error: no symbolGroup supplied');
    return;
  }

  if (!retrievalDate) {
    console.error('getDividendsforDate error: no retrievalDate supplied');
    return;
  }

  if (!symbolLookup) {
    console.error('getDividendsforDate error: no symbolLookup supplied');
    return;
  }


  // Retrieve dividends
  let dividendEndDate = utils.returnDateAsString(retrievalDate);
  let dividendStartDate = utils.dateAdd(dividendEndDate, 'years', -1);
  dividendStartDate = utils.dateAdd(dividendStartDate, 'days', 1);
  let dividends = awaitify(retrieveDividends(symbolGroup,
    dividendStartDate, dividendEndDate));

  // Convert dividend results array to results object
  let dividendResults = {};

  // Single company result is returned as an array
  if (Array.isArray(dividends) && dividends.length > 0) {
    // Multiple  symbols returned
    dividends.forEach((dividendRecord) => {
      processDividend(dividendRecord);
    });
  } else {
    // Multi-company result is returned as an object
    Object.keys(dividends).forEach((dividendCompany) => {
      dividends[dividendCompany].forEach((dividendRecord) => {
        processDividend(dividendRecord);
      });
    });
  }

  /** Check if dividend record is the latest record for each symbol.  If it is
  *    pdate the recorded dividend
  * @param {Object} dividendRecord record to process
  */
  function processDividend(dividendRecord) {
    let symbol = symbolLookup[dividendRecord.symbol];
    let exDividendDate = utils.returnDateAsString(dividendRecord['date']);
    let exDividendPayout = dividendRecord['dividends'];

    /* Check if symbol has already been recorded, or recorded with an earlier
      date */
    if (!dividendResults[symbol] ||
      dividendResults[symbol]['exDividendDate'] < exDividendDate)
      dividendResults[symbol] = {};
    dividendResults[symbol]['exDividendDate'] = exDividendDate;
    dividendResults[symbol]['exDividendPayout'] = exDividendPayout;
  }

  return dividendResults;
});

/**  Executes the company metrics update to add metrics information
*  after quotes are retrieved
* @param {Object} recLimits   (optional) - specifies where to start and stop
*     the operation, in form of:
*     {
        startRec: 100,
        endRec, 199,
*      }
*     This would process from company record index 100 to 199
*/
let executeMetricsUpdate = asyncify(function(recLimits) {
  // Get current dat based on last index quote for All Ordinaries
  let metricsDate = awaitify(getCurrentExecutionDate());
  let companies = [];
  let symbolResult = awaitify(setupSymbols());

  Object.keys(symbolResult.companyLookup).forEach((yahooSymbol) => {
    companies.push(symbolResult.companyLookup[yahooSymbol]);
  });

  if (metricsDate) {
    awaitify(updateQuotesWithMetrics(companies, metricsDate, recLimits || {}));
  }
});


module.exports = {
  setupSymbols: setupSymbols,
  updateQuotesWithMetrics: updateQuotesWithMetrics,
  writeIndexQuote: writeIndexQuote,
  writeCompanyQuoteData: writeCompanyQuoteData,
  returnIndexDataForDate: returnIndexDataForDate,
  executeFinancialIndicators: executeFinancialIndicators,
  executeCompanyMetrics: executeCompanyMetrics,
  executeIndexQuoteRetrieval: executeIndexQuoteRetrieval,
  executeCompanyQuoteRetrieval: executeCompanyQuoteRetrieval,
  executeMetricsUpdate: executeMetricsUpdate,
  retrieveDividends: retrieveDividends,
};
