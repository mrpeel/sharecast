'use strict';

const yahooFinance = require('yahoo-finance');
const utils = require('./utils');
const dynamodb = require('./dynamodb');
// const retrieveData = require('./dynamo-retrieve-share-data');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const moment = require('moment-timezone');
const sns = require('./publish-sns');
const snsArn = 'arn:aws:sns:ap-southeast-2:815588223950:lambda-activity';
const aws = require('aws-sdk');

const lambda = new aws.Lambda({
  region: 'ap-southeast-2',
});

/**  Reload all historical adjusted prices for a symbol
*    Retrieves the complete history of a symbol and updates the adjustedPrice -
*     triggered by a stock / split or other adjustment which causes historical
*     adjusted prices to be updated
* @param {Object} params only populated for continuing a previous lookup {
  symbol - symbol retrieved
  results - results
}
*/
let reloadQuote = asyncify(function(params) {
  let symbol;
  let symbolYear;
  try {
    let results = [];
    let yahooSymbol;
    let startDate;
    let endDate;

    let t0 = new Date();

    if (params.results && params.symbol && params.symbolYear) {
      results = params.results;
      symbol = params.symbol;
      symbolYear = params.symbolYear;
    } else if (!params.results && !params.symbol && !params.symbolYear) {
      // Retrieve next value from quoteReload table
      let scanDetails = {
        tableName: 'quoteReload',
        limit: 1,
      };

      let reloadResults = awaitify(dynamodb.scanTable(scanDetails));

      if (reloadResults.length) {
        symbolYear = reloadResults[0]['symbolYear'];
        symbol = reloadResults[0]['symbol'];
        yahooSymbol = reloadResults[0]['yahooSymbol'];
        startDate = reloadResults[0]['startDate'];
        endDate = reloadResults[0]['endDate'];
      } else {
        try {
          awaitify(
            sns.publishMsg(snsArn,
              `All reloadQuote value processing finished`, 'reloadQuote finished'));
        } catch (err) {}
        return true;
      }

      // Load results from yahoo
      let fullResults = awaitify(retrieveHistory(yahooSymbol,
        utils.returnDateAsString(startDate, 'YYYY-MM-DD'),
        utils.returnDateAsString(endDate, 'YYYY-MM-DD')));

      // Strip results to core values
      fullResults.forEach((resultVal) => {
        // Push the stripped down value to results
        results.push({
          'date': utils.returnDateAsString(resultVal['date']),
          'adjClose': resultVal['adjClose'],
        });
      });
    } else {
      console.error(`Inconsistent params: ${JSON.stringify(params)}`);
      return;
    }


    while (results.length) {
      let result = results.shift(1);
      result.adjustedPrice = result.adjClose;
      result.symbol = symbol;
      awaitify(updateAdjustedPrice(result));

      let t1 = new Date();

      if (utils.dateDiff(t0, t1, 'seconds') > 250) {
        break;
      }
    }

    /*  If results is not empty, more procesing is required
         re-invoke lambda with remaining results */
    if (results.length) {
      let reinvokeEvent = {
        'symbolYear': symbolYear,
        'symbol': symbol,
        'yahooSymbol': yahooSymbol,
        'results': results,
      };

      let description = `Re-invoking reloadQuote:  ${symbol} - ` +
        `${results.length} records`;

      awaitify(invokeLambda('reloadQuote', reinvokeEvent, description));
    } else {
      // Delete reload record
      let deleteDetails = {
        tableName: 'quoteReload',
        key: {
          'symbolYear': symbolYear,
        },
      };

      awaitify(dynamodb.deleteRecord(deleteDetails));

      // Invoke lambda again
      let description = `Invoking next reloadQuote`;

      awaitify(invokeLambda('reloadQuote', {}, description));
    }
    return true;
  } catch (err) {
    try {
      awaitify(
        sns.publishMsg(snsArn,
          'reloadQuote failed.  Error: ' + JSON.stringify(err),
          'reloadQuote failed'));
    } catch (err) {}

    console.log('reloadQuote failed while processing: ', symbolYear);
    console.log(err);
  }
});


let retrieveHistory = function(symbol, startDate, endDate) {
  return new Promise(function(resolve, reject) {
    let historyOptions = {
      from: startDate,
      to: endDate,
      symbol: symbol,
    };

    yahooFinance.historical(historyOptions).then(function(result) {
      resolve(result);
    }).catch(function(err) {
      reject(err);
    });
  });
};


/**  Write a conpany quote adjusted price update to dynamodb
* @param {Object} quoteData - the company quote to write
*/
let updateAdjustedPrice = asyncify(function(quoteData) {
  // If unexpected r3cords come back with no trade data, skop them
  if (!quoteData['date']) {
    return;
  }

  let updateDetails = {
    tableName: 'companyQuotes',
  };

  quoteData.quoteDate = utils.returnDateAsString(quoteData['date']);
  if (!quoteData.adjustedPrice) {
    return;
  }

  updateDetails.key = {
    symbol: quoteData.symbol,
    quoteDate: quoteData.quoteDate,
  };

  let updateExpression;
  let expressionAttributeValues = {};
  let expressionAttributeNames = {};

  // Separately set-up creeated attributed values and names
  expressionAttributeValues[':updated'] = moment().tz('Australia/Sydney').format();
  expressionAttributeNames['#updated'] = 'updated';

  expressionAttributeValues[':adjustedPrice'] = quoteData['adjustedPrice'];
  expressionAttributeNames['#adjustedPrice'] = 'adjustedPrice';

  updateExpression = 'set #adjustedPrice = :adjustedPrice';

  updateDetails.updateExpression = updateExpression;
  updateDetails.expressionAttributeValues = expressionAttributeValues;
  updateDetails.expressionAttributeNames = expressionAttributeNames;

  return awaitify(dynamodb.updateRecord(updateDetails));
});

let invokeLambda = function(lambdaName, event, description) {
  return new Promise(function(resolve, reject) {
    console.log(`Invoking lambda ${lambdaName} with event: ${JSON.stringify(event)}`);
    if (description) {
      console.log(description);
    }

    /*
    // Re-invoke function to simulate lambda
    retrieveAdjustedHistoryData(event);

    resolve(true);
    return; */

    lambda.invoke({
      FunctionName: lambdaName,
      InvocationType: 'Event',
      Payload: JSON.stringify(event, null, 2),
    }, function(err, data) {
      if (err) {
        reject(err);
      } else {
        console.log(`Function ${lambdaName} executed with event: `,
          `${JSON.stringify(event)}`);
        resolve(true);
      }
    });
  });
};


module.exports = {
  reloadQuote: reloadQuote,
};

// dynamodb.setLocalAccessConfig();
// reloadQuote({});
