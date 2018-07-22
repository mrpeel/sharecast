'use strict';

const utils = require('./utils');
const fetch = require('node-fetch');
// const asyncify = require('asyncawait/async');
// const awaitify = require('asyncawait/await');
const quoteURL =
    'https://query2.finance.yahoo.com/v10/finance/quoteSummary/$SYMBOL?formatted=false&modules=summaryDetail,financialData,earnings,price,defaultKeyStatistics,calendarEvents&corsDomain=finance.yahoo.com';
const dividendURL =
    'https://query1.finance.yahoo.com/v8/finance/chart/JBH.AX?symbol=$SYMBOL&events=div&period1=$STARTDATE&period2=$ENDDATE&interval=3mo';
const mapping = require('./yahoo-finance-mapping.json');

/**
 * Returns the current quote for a symbol
 * @param {String} symbol yahoo symbol name in form JBH.AX
 * @return {Object}  An object with the symbol quote details
 */
let retrieveQuote = function(symbol) {
    return new Promise(function(resolve, reject) {
        let fetchURL = quoteURL.replace(/\$SYMBOL/, symbol);

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
        let fetchURL = dividendURL.replace(/\$SYMBOL/, symbol);
        fetchURL = fetchURL.replace(/\$STARTDATE/, startDate);
        fetchURL = fetchURL.replace(/\$ENDDATE/, endDate);
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
                        let dividendDate = new Date(
                            dividendsObject[dividendKey].date * 1000
                        );
                        let dividendAmount = dividendsObject[dividendKey].amount;
                        dividends.push({
                            symbol: symbol,
                            dividendDate: utils.returnDateAsString(dividendDate),
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
                        let dateVal = new Date(yahooQuote[module][field] * 1000);
                        yahooQuote[module][field] = utils.returnDateAsString(dateVal);
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


module.exports = {
    getQuotes: getQuotes,
    getDividends: getDividends,
};

getQuotes({
    'symbols': ['JBH.AX', '123.AX'],
    'type': 'company',
}).then((quoteResults) => {
    console.log(`Number of succesfull results: ${JSON.stringify(quoteResults.results.length)}`);
    console.log(`Number of errors: ${quoteResults.errorCount}`);
    console.log(JSON.stringify(quoteResults.results));
    return getQuotes({
        'symbols': ['^AXJO', '^MADEUP'],
        'type': 'index',
    });
}).then((quoteResults) => {
    console.log(`Number of succesfull results: ${JSON.stringify(quoteResults.results.length)}`);
    console.log(`Number of errors: ${quoteResults.errorCount}`);
    console.log(JSON.stringify(quoteResults.results));
    return getDividends({
        'symbols': ['JBH.AX', '123.AX'],
        'startDate': '2017-07-31',
        'endDate': '2018-07-31',
    });
}).then((dividendResults) => {
    console.log(`Number of succesfull results: ${JSON.stringify(dividendResults.results.length)}`);
    console.log(`Number of errors: ${dividendResults.errorCount}`);
    console.log(JSON.stringify(dividendResults.results));
}).catch((err) => {
    console.error(err);
});
