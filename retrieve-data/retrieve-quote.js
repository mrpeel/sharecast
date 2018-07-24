'use strict';

const utils = require('./utils');
const fetch = require('node-fetch');
// const asyncify = require('asyncawait/async');
// const awaitify = require('asyncawait/await');
const quoteURL =
    'https://query2.finance.yahoo.com/v10/finance/quoteSummary/$SYMBOL?formatted=false&modules=summaryDetail,financialData,earnings,price,defaultKeyStatistics,calendarEvents&corsDomain=finance.yahoo.com';
const dividendURL =
    'https://query1.finance.yahoo.com/v8/finance/chart/$SYMBOL?symbol=$SYMBOL&events=div&period1=$STARTDATE&period2=$ENDDATE&interval=3mo';
const historyURL =
    'https://query1.finance.yahoo.com/v8/finance/chart/$SYMBOL?symbol=$SYMBOL&period1=$STARTDATE&period2=$ENDDATE&interval=1d';
const mapping = require('./yahoo-finance-mapping.json');

/**
 * Returns the current quote for a symbol
 * @param {String} symbol yahoo symbol name in form JBH.AX
 * @return {Object}  An object with the symbol quote details
 */
let retrieveQuote = function(symbol) {
    return new Promise(function(resolve, reject) {
        let fetchURL = quoteURL.replace(/\$SYMBOL/g, symbol);

        fetch(fetchURL)
            .then((response) => {
                return response.json();
            })
            .then((responseJson) => {
                // Check that the basic elements are presents
                if (
                    responseJson.quoteSummary &&
                    responseJson.quoteSummary.result &&
                    responseJson.quoteSummary.result.length
                ) {
                    resolve(responseJson.quoteSummary.result[0]);
                } else if (
                    responseJson.quoteSummary &&
                    responseJson.quoteSummary.error &&
                    responseJson.quoteSummary.error.code
                ) {
                    reject(responseJson.quoteSummary.error);
                } else {
                    reject({
                        code: 'Empty result',
                        description: `Empty result received for symbol: ${symbol}`,
                    });
                }
            })
            .catch((err) => {
                reject(err);
            });
    });
};

/**
 * Returns the dividends for a symbol between two dates
 * @param {String} symbol yahoo symbol name in form JBH.AX
 * @param {Float} startDate period start in unix format
 * @param {Float} endDate period end in unix format
 * @return {Array} Array of dividend objects
 */
let retrieveDividends = function(symbol, startDate, endDate) {
    return new Promise(function(resolve, reject) {
        let fetchURL = dividendURL.replace(/\$SYMBOL/g, symbol);
        fetchURL = fetchURL.replace(/\$STARTDATE/g, startDate);
        fetchURL = fetchURL.replace(/\$ENDDATE/g, endDate);
        let dividends = [];

        fetch(fetchURL)
            .then((response) => {
                return response.json();
            })
            .then((responseJson) => {
                // Check that the basic elements are presents
                if (
                    responseJson.chart &&
                    responseJson.chart.result &&
                    responseJson.chart.result[0].events &&
                    responseJson.chart.result[0].events.dividends
                ) {
                    let dividendsObject = responseJson.chart.result[0].events.dividends;
                    Object.keys(dividendsObject).forEach((dividendKey) => {
                        let dividendDate = utils.returnDateAsString(dividendsObject[dividendKey].date * 1000);
                        let dividendAmount = dividendsObject[dividendKey].amount;
                        dividends.push({
                            symbol: symbol,
                            dividendDate: dividendDate,
                            dividendAmount: dividendAmount,
                        });
                    });
                } else if (
                    responseJson.chart &&
                    responseJson.chart.error &&
                    responseJson.chart.error.code
                ) {
                    reject(responseJson.chart.error);
                }

                resolve(dividends);
            })
            .catch((err) => {
                reject(err);
            });
    });
};

/**
 * Returns the history (close & adjusted close) for a symbol in a timer period
 * @param {String} symbol yahoo symbol name in form JBH.AX
 * @param {Float} startDate period start in unix format
 * @param {Float} endDate period end in unix format
 * @param {Boolean} includeOriginalClose if true this includes the original close and the adjusted close
 * @return {Array} Array of close / adjusted close objects
 */
let retrieveHistory = function(symbol, startDate, endDate, includeOriginalClose) {
    return new Promise(function(resolve, reject) {
        let fetchURL = historyURL.replace(/\$SYMBOL/g, symbol);
        fetchURL = fetchURL.replace(/\$STARTDATE/g, startDate);
        fetchURL = fetchURL.replace(/\$ENDDATE/g, endDate);

        fetch(fetchURL)
            .then((response) => {
                return response.json();
            })
            .then((responseJson) => {
                // Check that the basic elements are presents
                if (
                    responseJson.chart &&
                    responseJson.chart.result &&
                    responseJson.chart.result[0].timestamp &&
                    responseJson.chart.result[0].indicators &&
                    responseJson.chart.result[0].indicators.quote &&
                    responseJson.chart.result[0].indicators.quote[0].close &&
                    responseJson.chart.result[0].indicators.adjclose &&
                    responseJson.chart.result[0].indicators.adjclose[0].adjclose
                ) {
                    // Check record lengths
                    let timestampRecords = responseJson.chart.result[0].timestamp;
                    let closeRecords = responseJson.chart.result[0].indicators.quote[0].close;
                    let adjustedCloseRecords = responseJson.chart.result[0].indicators.adjclose[0].adjclose;
                    let historyResult = [];

                    console.log(`History returned for ${symbol}. ${timestampRecords.length} timestamp records, ${closeRecords.length} close records, ${adjustedCloseRecords.length} close records`);

                    if (timestampRecords.length == closeRecords.length && timestampRecords.length == adjustedCloseRecords.length) {
                        for (let hc = 0; hc < timestampRecords.length; hc++) {
                            // Always include date and adjusted close
                            let pushRecord = {
                                'date': utils.returnDateAsString(timestampRecords[hc] * 1000),
                                'adjClose': adjustedCloseRecords[hc],
                            };

                            // If including the original close, add to record
                            if (includeOriginalClose) {
                                pushRecord.close = closeRecords[hc];
                            }
                            historyResult.push(pushRecord);
                        }
                        resolve(historyResult);
                    } else {
                        reject({
                            code: 'Inconcistent result',
                            description: `Different number of timestamp, close & adjusted close records received for symbol: ${symbol}`,
                        });
                    }
                } else if (
                    responseJson.chart &&
                    responseJson.chart.error &&
                    responseJson.chart.error.code
                ) {
                    reject(responseJson.chart.error);
                } else {
                    reject({
                        code: 'Incomplete result',
                        description: `Incomplete result received for symbol: ${symbol}`,
                    });
                }
            })
            .catch((err) => {
                reject(err);
            });
    });
};

/**  Converts a yahoo quote into the name mapping required and turns numeric dates into YYY-MM-DD
 * @param {Object} yahooQuote - the yahoo quote result
 * @param {Boolean} isCompanySymbol - whether this a company symbol (false = index symbol)
 * @return {Object} the result object with name mapping and date fields converted
 */
let transformQuote = function(yahooQuote, isCompanySymbol = true) {
    let dateMapping = mapping.dateFields;
    let transformedQuote = {};
    try {
        // Work through date fields and convert
        Object.keys(dateMapping).forEach((module) => {
            // Check that the module with date exists in the quote
            if (yahooQuote[module]) {
                dateMapping[module].forEach((field) => {
                    // Check the field with the date exists in the module
                    if (yahooQuote[module][field]) {
                        // Perform conversion
                        yahooQuote[module][field] = utils.returnDateAsString(yahooQuote[module][field] * 1000);
                    }
                });
            }
        });

        let mappingKey = isCompanySymbol ? 'company' : 'index';
        let fieldMapping = mapping[mappingKey];

        Object.keys(fieldMapping).forEach((field) => {
            // Extract the field parts - separate by '.'
            let fieldParts = fieldMapping[field].split('.');
            let moduleName = fieldParts[0];
            let fieldName = fieldParts[1];

            // Check field is present in object and copy with new name
            if (yahooQuote[moduleName][fieldName]) {
                transformedQuote[field] = yahooQuote[moduleName][fieldName];
            }
        });

        // Calculate 52 week changes
        let quotePrice = isCompanySymbol ?
            transformedQuote.lastTradePriceOnly :
            transformedQuote.previousClose;

        if (quotePrice) {
            if (transformedQuote['52WeekHigh']) {
                transformedQuote['changeFrom52WeekHigh'] =
                    quotePrice - transformedQuote['52WeekHigh'];
                transformedQuote['percebtChangeFrom52WeekHigh'] =
                    (transformedQuote - transformedQuote['52WeekHigh']) /
                    transformedQuote['52WeekHigh'];
            } else {
                transformedQuote['percebtChangeFrom52WeekHigh'] = 0;
                transformedQuote['changeFrom52WeekHigh'] = 0;
            }

            if (transformedQuote['52WeekLow']) {
                transformedQuote['changeFrom52WeekLow'] =
                    quotePrice - transformedQuote['52WeekLow'];
                transformedQuote['percentChangeFrom52WeekLow'] =
                    (quotePrice - transformedQuote['52WeekLow']) /
                    transformedQuote['52WeekLow'];
            } else {
                transformedQuote['percentChangeFrom52WeekLow'] = 0;
                transformedQuote['changeFrom52WeekLow'] = 0;
            }
        }

        return transformedQuote;
    } catch (err) {
        console.error('Mapping error');
    }
};

/**
 * Return and transform a quote
 * @param {Object} retrieveOptions an object in the form:
 *  {
 *     'symbols': [] an array of yahoo symbol names in form JBH.AX
 *     'type': 'index' or 'company'
 *  }
 * @return {Array}  An array of objects with the symbol quote details
 */
let getQuotes = async function(retrieveOptions) {
    try {
        if (!retrieveOptions || !retrieveOptions.symbols || !retrieveOptions.type) {
            throw new Error(
                `getQuotes missing symbols or type property: ${JSON.stringify(retrieveOptions)}`
            );
        }

        let symbolResults = [];
        let symbolErrors = 0;
        let isCompanySymbol = retrieveOptions.type === 'index' ? false : true;

        for (let sc = 0; sc < retrieveOptions.symbols.length; sc++) {
            let symbol = retrieveOptions.symbols[sc];
            try {
                let rawQuote = await retrieveQuote(symbol);
                let symbolQuote = transformQuote(rawQuote, isCompanySymbol);

                symbolResults.push(symbolQuote);
                console.log(`${symbol} quote retrieved`);
            } catch (err) {
                console.log(`${symbol} error: ${JSON.stringify(err)}`);
                symbolErrors++;
            }
        }

        return {
            results: symbolResults,
            errorCount: symbolErrors,
        };
    } catch (err) {
        throw (err);
    }
};

/**
 * Get dividends for a set of symbols for a time period
 * @param {Object} retrieveOptions an object in the form:
 *  {
 *     'symbols': [],  an array yahoo symbol name in form JBH.AX
 *     'startDate': period start in format YYYY-MM-DD
 *     'endDate': period end in format YYYY-MM-DD
 *  }
 * @return {Array}  An array of objects with the symbol dividends
 */
let getDividends = async function(retrieveOptions) {
    try {
        if (!retrieveOptions ||
            !retrieveOptions.symbols ||
            !retrieveOptions.startDate ||
            !retrieveOptions.endDate
        ) {
            throw new Error(
                `getDividends missing symbols, startDate or endDate property: ${JSON.stringify(retrieveOptions)}`
            );
        }
        let startDate = utils.returnDateAsUnix(retrieveOptions.startDate);
        let endDate = utils.returnDateAsUnix(retrieveOptions.endDate);
        let dividendResults = [];
        let symbolErrors = 0;
        for (let sc = 0; sc < retrieveOptions.symbols.length; sc++) {
            let symbol = retrieveOptions.symbols[sc];
            try {
                let symbolDividends = await retrieveDividends(
                    symbol,
                    startDate,
                    endDate
                );
                dividendResults.push({
                    symbol: symbol,
                    dividends: symbolDividends,
                });
                console.log(`${symbol} retrieved ${symbolDividends.length} dividends records`);
            } catch (err) {
                console.log(`${symbol} error: ${JSON.stringify(err)}`);
                symbolErrors++;
            }
        }

        return {
            results: dividendResults,
            errorCount: symbolErrors,
        };
    } catch (err) {
        throw (err);
    }
};

/**
 * Checks whether there is a difference between the close and adjusted close
 *   for a symbol in a time period
 * @param {Object} retrieveOptions an object in the form:
 *  {
 *     'symbol': yahoo symbol name in form JBH.AX
 *     'startDate': period start in format YYYY-MM-DD
 *     'endDate': period end in format YYYY-MM-DD
 *  }
 * @return {Boolean}  whether any close price has been adjusted
 */
let checkForAdjustedPrice = async function(retrieveOptions) {
    try {
        if (!retrieveOptions ||
            !retrieveOptions.symbol ||
            !retrieveOptions.startDate ||
            !retrieveOptions.endDate
        ) {
            throw new Error(
                `getDividends missing symbol, startDate or endDate property: ${JSON.stringify(retrieveOptions)}`
            );
        }
        let startDate = utils.returnDateAsUnix(retrieveOptions.startDate);
        let endDate = utils.returnDateAsUnix(retrieveOptions.endDate);
        let symbol = retrieveOptions.symbol;
        let symbolHistory = await retrieveHistory(
            symbol,
            startDate,
            endDate,
            true
        );
        console.log(`${symbol} retrieved ${symbolHistory.length} history records`);

        // Check for records with close != adjClose
        let adjustedResult = symbolHistory.some((historyRecord) => {
            if (historyRecord.adjClose && historyRecord.close) {
                return (historyRecord.adjClose !== historyRecord.close);
            } else {
                return false;
            }
        });

        return adjustedResult;
    } catch (err) {
        console.log(`${retrieveOptions.symbol} error: ${JSON.stringify(err)}`);
        throw (err);
    }
};

/**
 * Checks whether there is a difference between the close and adjusted close
 *   for a symbol in a time period
 * @param {Object} retrieveOptions an object in the form:
 *  {
 *     'symbols': [],  an array yahoo symbol name in form JBH.AX
 *     'startDate': period start in format YYYY-MM-DD
 *     'endDate': period end in format YYYY-MM-DD
 *  }
 * @return {Boolean}  whether any close price has been adjusted
 */
let getAdjustedPrices = async function(retrieveOptions) {
    try {
        if (!retrieveOptions ||
            !retrieveOptions.symbols ||
            !retrieveOptions.startDate ||
            !retrieveOptions.endDate
        ) {
            throw new Error(
                `getAdjustedPrices missing symbols, startDate or endDate property: ${JSON.stringify(retrieveOptions)}`
            );
        }
        let startDate = utils.returnDateAsUnix(retrieveOptions.startDate);
        let endDate = utils.returnDateAsUnix(retrieveOptions.endDate);
        let historyResults = [];
        let symbolErrors = 0;
        for (let sc = 0; sc < retrieveOptions.symbols.length; sc++) {
            let symbol = retrieveOptions.symbols[sc];
            try {
                let symbolHistory = await retrieveHistory(
                    symbol,
                    startDate,
                    endDate,
                    false
                );
                historyResults.push({
                    symbol: symbol,
                    history: symbolHistory,
                });
                console.log(`${symbol} retrieved ${symbolHistory.length} history records`);
            } catch (err) {
                console.log(`${symbol} error: ${JSON.stringify(err)}`);
                symbolErrors++;
            }
        }

        return {
            results: historyResults,
            errorCount: symbolErrors,
        };
    } catch (err) {
        throw (err);
    }
};

module.exports = {
    getQuotes: getQuotes,
    getDividends: getDividends,
    checkForAdjustedPrice: checkForAdjustedPrice,
    getAdjustedPrices: getAdjustedPrices,
};

checkForAdjustedPrice({
    'symbol': 'JBH.AX',
    'startDate': '2018-07-18',
    'endDate': '2018-07-25',
}).then((adjustedResults) => {
    console.log(JSON.stringify(adjustedResults));
    return getAdjustedPrices({
        'symbols': ['JBH.AX'],
        'startDate': '2007-01-01',
        'endDate': '2018-07-30',
    });
}).then((historyResults) => {
    // console.log(JSON.stringify(historyResults.results, null, 2));
    historyResults.results.forEach((result) => {
        console.log(`${result.symbol}: ${result.history.length} results`);
    });
    console.log(`${historyResults.errorCount} errors`);
}).catch((err) => {
    console.error(err);
});

// getQuotes({
//     'symbols': ['JBH.AX', '123.AX'],
//     'type': 'company',
// }).then((quoteResults) => {
//     console.log(`Number of succesfull results: ${JSON.stringify(quoteResults.results.length)}`);
//     console.log(`Number of errors: ${quoteResults.errorCount}`);
//     console.log(JSON.stringify(quoteResults.results));
//     return getQuotes({
//         'symbols': ['^AXJO', '^MADEUP'],
//         'type': 'index',
//     });
// }).then((quoteResults) => {
//     console.log(`Number of succesfull results: ${JSON.stringify(quoteResults.results.length)}`);
//     console.log(`Number of errors: ${quoteResults.errorCount}`);
//     console.log(JSON.stringify(quoteResults.results));
//     return getDividends({
//         'symbols': ['JBH.AX', '123.AX'],
//         'startDate': '2017-07-31',
//         'endDate': '2018-07-31',
//     });
// }).then((dividendResults) => {
//     console.log(`Number of succesfull results: ${JSON.stringify(dividendResults.results.length)}`);
//     console.log(`Number of errors: ${dividendResults.errorCount}`);
//     console.log(JSON.stringify(dividendResults.results));
// }).catch((err) => {
//     console.error(err);
// });
