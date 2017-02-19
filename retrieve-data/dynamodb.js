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
*  primaryKey: [ (optional for conditional check )
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

    // Add created timestamp if missing
    if (!item['created']) {
      item['created'] = moment().tz('Australia/Sydney').format();
    }

    let params = {
      TableName: insertDetails.tableName,
      Item: item,
    };

    if (insertDetails.primaryKey) {
      params.ConditionExpression = 'attribute_not_exists(' +
        insertDetails.primaryKey[0] + ')';
    }

    console.log('Put table request: ', JSON.stringify(params));

    client.put(params, function(err, data) {
      if (err && err.code === 'ConditionalCheckFailedException') {
        console.error('Skipping add to ', insertDetails.tableName, ': ',
          JSON.stringify(insertDetails.primaryKey), ' already exists.',
          ', values: ', JSON.stringify(insertDetails.values));
        resolve({
          result: 'skipped',
          message: 'Skipping add to ' + insertDetails.tableName + ': ' +
            JSON.stringify(insertDetails.primaryKey) + ' already exists.' +
            ', values: ' + JSON.stringify(insertDetails.values),
        });
      } else if (err) {
        console.error('Unable to add to ', insertDetails.tableName,
          ', values: ', JSON.stringify(insertDetails.values), '. Error JSON:',
          JSON.stringify(err, null, 2));
        reject({
          result: 'failed',
          message: JSON.stringify(err, null, 2),
        });
      } else {
        console.log('PutItem to ', insertDetails.tableName, ' succeeded.');
        resolve({
          result: 'inserted',
          data: JSON.stringify(insertDetails.values),
        });
      }
    });
  });
};

/** Query a table and return any matching records
* @param {Object} queryDetails - an object with all the details for insert
* queryDetails = {
*  tableName: 'companies',
*  indexName: 'company-created-index' (optional secondary index to use),
*  keyConditionExpression: 'symbol = :symbol',
*  expressionAttributeValues: {
*  ':symbol': 'AAD',
*  },
*  limit: 1, (optional query limit)
*  reverseOrder: false, (optional to reverse sort order)
*  projectionExpression: 'symbol, quoteDate',(optional expression to limit
*                      the fields returned)
* };
@return {Promise} which resolves with:
*   array of data items
*/
let queryTable = function(queryDetails) {
  return new Promise(function(resolve, reject) {
    let params = {
      TableName: queryDetails.tableName,
      KeyConditionExpression: queryDetails.keyConditionExpression,
      ExpressionAttributeValues: queryDetails.expressionAttributeValues,
    };

    if (queryDetails.limit) {
      params.Limit = queryDetails.limit;
    }

    if (queryDetails.indexName) {
      params.IndexName = queryDetails.indexName;
    }

    if (queryDetails.reverseOrder) {
      params.ScanIndexForward = false;
    } else {
      params.ScanIndexForward = true;
    }

    if (queryDetails.projectionExpression) {
      params.ProjectionExpression = queryDetails.projectionExpression;
    }

    console.log('Query table request: ', JSON.stringify(params));

    client.query(params, function(err, data) {
      if (err) {
        console.error('Unable to query ', queryDetails.tableName,
          '. Error:', JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log('Query table ', queryDetails.tableName, ' succeeded.  ',
          data.Items.length, ' returned.');
        resolve(data.Items || []);
      }
    });
  });
};

/** Scan a table and return any matching records
* @param {Object} scanDetails - an object with all the details for insert
* queryDetails = {
*  tableName: 'companies',
*  filterExpression: 'symbol = :symbol',
*  expressionAttributeValues: {
*  ':symbol': 'AAD',
*  },
*  limit: 1, (optional query limit)
*  projectionExpression: 'symbol, quoteDate',(optional expression to limit
*                      the fields returned)
* };
@return {Promise} which resolves with:
*   array of data items
*/
let scanTable = function(scanDetails) {
  return new Promise(function(resolve, reject) {
    let params = {
      TableName: scanDetails.tableName,
      FilterExpression: scanDetails.filterExpression,
      ExpressionAttributeValues: scanDetails.expressionAttributeValues,
    };

    if (scanDetails.limit) {
      params.Limit = scanDetails.limit;
    }

    if (queryDetails.projectionExpression) {
      params.ProjectionExpression = queryDetails.projectionExpression;
    }

    console.log('Scan table request: ', JSON.stringify(params));


    client.scan(params, (err, data) => {
      if (err) {
        console.error('Unable to scan ', scanDetails.tableName,
          '. Error:', JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log('Scan table ', scanDetails.tableName, ' succeeded.',
          data.Items.length, ' returned.');
        resolve(data.Items || []);
      }
    });
  });
};

/** Scan a table and return all records
* @param {Object} tableDetails - an object with all the details
* tableDetails = {
*  tableName: 'companies',
*  reverseOrder: false, (optional to reverse sort order)
*  projectionExpression: 'symbol, quoteDate',(optional expression to limit
*                      the fields returned)
* };
@return {Promise} which resolves with:
*   array of data items
*/
let getTable = function(tableDetails) {
  return new Promise(function(resolve, reject) {
    let params = {
      TableName: tableDetails.tableName,
    };


    if (tableDetails.reverseOrder) {
      params.ScanIndexForward = false;
    }

    if (tableDetails.projectionExpression) {
      params.ProjectionExpression = tableDetails.projectionExpression;
    }

    console.log('Get table request: ', JSON.stringify(params));

    client.scan(params, function(err, data) {
      if (err) {
        console.error('Unable to get table  ', tableDetails.tableName,
          '. Error:', JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log('Get table ', tableDetails.tableName, ' succeeded.',
          data.Items.length, ' returned.');
        resolve(data.Items || []);
      }
    });
  });
};

/** Update a record in a table
* @param {Object} updateDetails - an object with all the details
* tableDetails = {
*  tableName: 'companies',
*  key: {
            symbol: id,
            quoteDate: ''
        },
]
*  updateExpression: 'set #sentTime = :sentTime, sentLog = :sentLog',
*  expressionAttributeValues: {
    ':sentTime': moment().tz('Australia/Sydney').format(),
    ':sentLog': result
  },
  *  expressionAttributeNames: (optional) {
      '#sentTime': 'SentTime',
    },
* };
@return {Promise} which resolves with:
*   array of data items
*/
let updateRecord = function(updateDetails) {
  return new Promise(function(resolve, reject) {
    let params = {
      TableName: updateDetails.tableName,
      Key: updateDetails.key,
      UpdateExpression: updateDetails.updateExpression,
      ExpressionAttributeValues: updateDetails.expressionAttributeValues,
    };

    if (updateDetails.expressionAttributeNames) {
      params.ExpressionAttributeNames = updateDetails.expressionAttributeNames;
    }

    params.ReturnValues = 'UPDATED_NEW';

    console.log('Update table request: ', JSON.stringify(params));

    client.update(params, function(err, data) {
      if (err) {
        console.error('Unable to update table  ', updateDetails.tableName,
          '. Error:', JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log('Update table ', updateDetails.tableName, ' succeeded.');
        resolve(data || null);
      }
    });
  });
};

/* let testQuery = function() {
  let queryDetails = {
    tableName: 'financialIndicatorValues',
    indexName: 'symbol-created-index',
    keyConditionExpression: 'symbol = :symbol and created <= :created',
    expressionAttributeValues: {
      ':symbol': 'T1',
      ':created': '2017-05-02',
    },
    reverseOrder: true,
    limit: 1,
  };

  queryTable(queryDetails).then((result) => {
    console.log(JSON.stringify(result));
  }).catch((err) => {
    console.log(err);
  });
}; */

/* let testScan = function() {
  let scanDetails = {
    tableName: 'financialIndicatorValues',
    filterExpression: 'symbol = :symbol and created <= :created',
    expressionAttributeValues: {
      ':symbol': 'T1',
      ':created': '2017-05-02',
    },
    limit: 1,
  };

  scanTable(scanDetails).then((result) => {
    console.log(JSON.stringify(result));
  }).catch((err) => {
    console.log(err);
  });
}; */

// testQuery();

module.exports = {
  insertRecord: insertRecord,
  queryTable: queryTable,
  scanTable: scanTable,
  getTable: getTable,
  updateRecord: updateRecord,
};
