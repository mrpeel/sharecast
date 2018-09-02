'use strict';

const utils = require('./utils');
const dynamodb = require('./dynamodb');
const retrieveQuote = require('./retrieve-quote');
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
let reloadQuote = async function(params) {
  let symbol;
  let symbolYear;
  let resultStats;

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

      if (params.resultStats) {
        resultStats = params.resultStats;
      } else {
        resultStats = {
          updatedRecords: 0,
          errors: 0,
        };
      }

      let reloadResults = await dynamodb.scanTable(scanDetails);

      if (reloadResults.length) {
        symbolYear = reloadResults[0]['symbolYear'];
        symbol = reloadResults[0]['symbol'];
        yahooSymbol = reloadResults[0]['yahooSymbol'];
        startDate = reloadResults[0]['startDate'];
        endDate = reloadResults[0]['endDate'];
      } else {
        try {
          await sns.publishMsg(snsArn,
            `${JSON.stringify(resultStats, null, 2)}`, 'reloadQuote finished');
        } catch (err) {}
        return true;
      }

      // Delete reload record, so it isn't run multiple times
      let deleteDetails = {
        tableName: 'quoteReload',
        key: {
          'symbolYear': symbolYear,
        },
      };

      await dynamodb.deleteRecord(deleteDetails);

      // Load results from yahoo
      let fullResults = await retrieveQuote.getAdjustedPrices({
        'symbols': [yahooSymbol],
        'startDate': utils.returnDateAsString(startDate, 'YYYY-MM-DD'),
        'endDate': utils.returnDateAsString(endDate, 'YYYY-MM-DD'),
      });

      fullResults.results.forEach((result) => {
        console.log(`${result.symbol}: ${result.history.length} results`);
        // Loop through historical ajdusted close values
        result.history.forEach((historyRec) => {
          // Push the stripped down value to results if the value is present
          if (historyRec.adjClose) {
            results.push({
              'date': utils.returnDateAsString(historyRec.date),
              'adjClose': historyRec.adjClose,
            });
          }
        });
      });
      console.log(`${fullResults.errorCount} errors`);
    } else {
      console.error(`Inconsistent params: ${JSON.stringify(params)}`);
      return;
    }


    while (results.length) {
      let result = results.shift(1);
      result.adjustedPrice = result.adjClose;
      result.symbol = symbol;
      try {
        let updateResult = await updateAdjustedPrice(result);

        if (updateResult.error) {
          resultStats.errors++;
        } else {
          resultStats.updatedRecords++;
        }
      } catch (err) {
        console.log(`Error while processing ${symbol} for ${result.date}. ${err}`);
      }

      let t1 = new Date();

      if (utils.dateDiff(t0, t1, 'seconds') > (dynamodb.getExecutionMaxTime() - 40)) {
        await sns.publishMsg(snsArn,
          `${JSON.stringify(resultStats, null, 2)}`, 'reloadQuote Continuing');
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
        'resultStats': resultStats,
      };

      let description = `Re-invoking reloadQuote:  ${symbolYear} - ` +
        `${results.length} records`;

      await invokeLambda('reloadQuote', reinvokeEvent, description);
    } else {
      // Invoke lambda again
      let description = `Invoking next reloadQuote`;

      await invokeLambda('reloadQuote', {
        'resultStats': resultStats,
      }, description);
    }
    return true;
  } catch (err) {
    try {
      await
      sns.publishMsg(snsArn,
        'reloadQuote failed.  Error: ' + JSON.stringify(err, null, 2),
        'reloadQuote failed');
    } catch (err) {}

    console.log('reloadQuote failed while processing: ', symbolYear);
    console.log(err);
  }
};


/**  Write a conpany quote adjusted price update to dynamodb
 * @param {Object} quoteData - the company quote to write
 */
let updateAdjustedPrice = async function(quoteData) {
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

  return await dynamodb.updateRecord(updateDetails);
};

let invokeLambda = async function(lambdaName, event, description) {
  console.log(`Invoking lambda ${lambdaName} with event: ${JSON.stringify(event)}`);
  if (description) {
    console.log(description);
  }


  // Re-invoke function to simulate lambda
  // reloadQuote(event);
  // return true;

  try {
    await lambda.invoke({
      FunctionName: lambdaName,
      InvocationType: 'Event',
      Payload: JSON.stringify(event),
    }).promise();
    console.log(`Function ${lambdaName} executed with event: `,
      `${JSON.stringify(event, null, 2)}`);
    return true;
  } catch (err) {
    throw err;
  }
};


module.exports = {
  reloadQuote: reloadQuote,
};

// dynamodb.setLocalAccessConfig();
// reloadQuote({});
