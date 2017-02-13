'use strict';

const AWS = require('aws-sdk');
const moment = require('moment-timezone');

AWS.config.loadFromPath('../credentials/aws.json');

const client = new AWS.DynamoDB.DocumentClient();

/** Insert record into dynamodb if it doesn't already exist
* @param {Object} insertDetails - an object with all the details for insert
* insertDetails = {
*  tableName: 'companies',
*  values: {
*    companySymbol: '1AG',
*    companyName: 'Alterra Limited',
*    companySymbolYahoo: '1AG.AX',
*    companySymbolGoogle: '1AG:AX',
*    watchingCompany: false,
*  },
*  primaryKey: [
*    'companySymbol',
*  ],
* };
@return {Promise} which resolves with:
* {
*   result: skipped / inserted / failed
*   data: data inserted (optional)
*   message: error message (optional)
* }
*/
let insertRecord = function(insertDetails) {
  return new Promise(function(resolve, reject) {
    // Set-up item details for insert
    let item = {};

    Object.keys(insertDetails.values).forEach((valueKey) => {
      item[valueKey] = insertDetails.values[valueKey];
    });

    // Add created timestamp
    item['created'] = moment().tz('Australia/Sydney').format();

    let params = {
      TableName: insertDetails.tableName,
      Item: item,
      ConditionExpression: 'attribute_not_exists(' + insertDetails.primaryKey[0]
        + ')',
    };


    client.put(params, function(err, data) {
      if (err && err.code === 'ConditionalCheckFailedException') {
        console.error('Skipping add to ' + insertDetails.tableName + ': ',
          JSON.stringify(insertDetails.primaryKey), ' already exists.' +
          ', values: ', JSON.stringify(insertDetails.values));
        resolve({
          result: 'skipped',
          message: 'Skipping add to ' + insertDetails.tableName + ': ' +
            JSON.stringify(insertDetails.primaryKey) + ' already exists.' +
            ', values: ' + JSON.stringify(insertDetails.values),
        });
      } else if (err) {
        console.error('Unable to add to ' + insertDetails.tableName +
        ', values: ', JSON.stringify(insertDetails.values), '. Error JSON:',
          JSON.stringify(err, null, 2));
        reject({
          result: 'failed',
          message: JSON.stringify(err, null, 2),
        });
      } else {
        console.log('PutItem succeeded to ' + insertDetails.tableName +
          ', values: ', JSON.stringify(insertDetails.values));
        resolve({
          result: 'inserted',
          data: JSON.stringify(insertDetails.values),
        });
      }
    });
  });
};

/* let insertDetails = {
  tableName: 'companies',
  values: {
    companySymbol: '1AG',
    companyName: 'Alterra Limited',
    companySymbolYahoo: '1AG.AX',
    companySymbolGoogle: '1AG:AX',
    watchingCompany: false,
  },
  primaryKey: [
    'companySymbol',
  ],
}; */
/*
let insertDetails = {
  tableName: 'companyMetrics',
  values: {
    'companySymbol': '1AJ',
    'metricsDate': '2017-02-10',
    'EPS': 0.32,
    'QuoteLast': 7.55,
    'Price200DayAverage': 7.29,
    'Price52WeekPercChange': 3.59,
    'PriceToBook': 5.17,
    'MarketCap': 178770000,
    'PE': 23.45,
  },
  primaryKey: [
    'companySymbol', 'metricsDate',
  ],
};

insertRecord(insertDetails)
  .then((result) => {
    console.log(result);
  }).catch((err) => {
  console.log(err.message);
}); */


module.exports = {
  insertRecord: insertRecord,
};
