'use strict';

const AWS = require('aws-sdk');
const moment = require('moment-timezone');


AWS.config.loadFromPath('../credentials/aws.json');

const client = new AWS.DynamoDB.DocumentClient();
let dynamoTables = {};

/** Gets all the dynamo tables and records their provisioned capacity
*/
let getTableInfo = function() {
  let db = new AWS.DynamoDB();

  db.listTables({}, function(err, data) {
    if (err) {
      console.log(err);
    } else {
      data.TableNames.forEach((tableName) => {
        db.describeTable({
          TableName: tableName,
        }, function(err, data) {
          if (err) {
            console.log(err);
          } else {
            dynamoTables[tableName] = {};
            dynamoTables[tableName]['readCapacity'] = data.Table.ProvisionedThroughput.ReadCapacityUnits;
            dynamoTables[tableName]['writeCapacity'] = data.Table.ProvisionedThroughput.WriteCapacityUnits;
          }
        });
      });
    }
  });
};

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
    let maxCapacity = 5; // default write capacity to 5

    // Set-up max write capacity
    if (dynamoTables[insertDetails.tableName] &&
      dynamoTables[insertDetails.tableName]['writeCapacity']) {
      maxCapacity = dynamoTables[insertDetails.tableName]['writeCapacity'];
    }

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
      ReturnConsumedCapacity: 'TOTAL',
    };

    if (insertDetails.primaryKey) {
      params.ConditionExpression = 'attribute_not_exists(' +
        insertDetails.primaryKey[0] + ')';
    }


    console.log('Put table request: ', JSON.stringify(params));

    client.put(params, function(err, data) {
      let timeout = 0;
      // Retrieved used capacity and check if it's over allowed
      if (data && data.ConsumedCapacity &&
        data.ConsumedCapacity.CapacityUnits > maxCapacity) {
        timeout = 1000 / data.ConsumedCapacity.CapacityUnits;
      }
      setTimeout(function() {
        if (err && err.code === 'ConditionalCheckFailedException') {
          console.log(`Skipping add to ${insertDetails.tableName} : `,
            JSON.stringify(insertDetails.primaryKey), ' already exists');
          resolve({
            result: 'skipped',
            message: `Skipping add to ${insertDetails.tableName} : ` +
              JSON.stringify(insertDetails.primaryKey) + ' already exists.',
          });
        } else if (err) {
          console.error(`Unable to add to ${insertDetails.tableName} values: `,
            '. Error JSON:', JSON.stringify(err, null, 2));
          reject({
            result: 'failed',
            message: JSON.stringify(err, null, 2),
          });
        } else {
          console.log(`PutItem to ${insertDetails.tableName} succeeded.`);
          resolve({
            result: 'inserted',
          });
        }
      }, timeout);
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
*  filterExpression: 'quoteDate <= :quoteDate' (optional),
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
    let queryDataItems = [];
    let params = {
      TableName: queryDetails.tableName,
      KeyConditionExpression: queryDetails.keyConditionExpression,
      ExpressionAttributeValues: queryDetails.expressionAttributeValues,
    };

    let maxCapacity = 5; // default read capacity to 5
    let resultsLimit;

    // Set-up max write capacity
    if (dynamoTables[queryDetails.tableName] &&
      dynamoTables[queryDetails.tableName]['readCapacity']) {
      maxCapacity = dynamoTables[queryDetails.tableName]['readCapacity'];
    }

    params.Limit = maxCapacity;

    if (queryDetails.limit) {
      resultsLimit = queryDetails.limit;
      if (resultsLimit < maxCapacity) {
        params.Limit = resultsLimit;
      }
    }

    if (queryDetails.indexName) {
      params.IndexName = queryDetails.indexName;
    }

    if (queryDetails.reverseOrder) {
      params.ScanIndexForward = false;
    } else {
      params.ScanIndexForward = true;
    }

    if (queryDetails.filterExpression) {
      params.FilterExpression = queryDetails.filterExpression;
    }


    if (queryDetails.projectionExpression) {
      params.ProjectionExpression = queryDetails.projectionExpression;
    }

    console.log('Query table request: ', JSON.stringify(params));

    let onQuery = function(err, data) {
      if (err) {
        console.error(`Unable to query ${queryDetails.tableName}. Error: `,
          JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log(`Query table ${queryDetails.tableName} succeeded.  `,
          `${data.Items.length} records returned.`);

        queryDataItems = queryDataItems.concat(data.Items);

        // continue scanning if we have more movies, because
        // scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey !== 'undefined' &&
          (!resultsLimit || queryDataItems.length < resultsLimit)) {
          console.log('Querying for more data...');
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          client.query(params, onQuery);
        } else {
          resolve(queryDataItems || []);
        }
      }
    };

    client.query(params, onQuery);
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
    let scanDataItems = [];
    let params = {
      TableName: scanDetails.tableName,
      FilterExpression: scanDetails.filterExpression,
      ExpressionAttributeValues: scanDetails.expressionAttributeValues,
    };

    let maxCapacity = 5; // default read capacity to 5
    let resultsLimit;

    // Set-up max write capacity
    if (dynamoTables[scanDetails.tableName] &&
      dynamoTables[scanDetails.tableName]['readCapacity']) {
      maxCapacity = dynamoTables[scanDetails.tableName]['readCapacity'];
    }

    params.Limit = maxCapacity;

    if (scanDetails.limit) {
      resultsLimit = scanDetails.limit;
      if (resultsLimit < maxCapacity) {
        params.Limit = resultsLimit;
      }
    }

    if (scanDetails.projectionExpression) {
      params.ProjectionExpression = scanDetails.projectionExpression;
    }

    console.log('Scan table request: ', JSON.stringify(params));

    let onScan = function(err, data) {
      if (err) {
        console.error(`Unable to scan ${scanDetails.tableName}. Error: `,
          JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log(`Scan table ${scanDetails.tableName} succeeded. `,
          `${data.Items.length} records returned.`);

        scanDataItems = scanDataItems.concat(data.Items);

        // continue scanning if we have more movies, because
        // scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey !== 'undefined' &&
          (!resultsLimit || scanDataItems.length < resultsLimit)) {
          console.log('Scanning for more data...');
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          client.scan(params, onScan);
        } else {
          resolve(scanDataItems || []);
        }
      }
    };

    client.scan(params, onScan);
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

    let scanDataItems = [];

    let maxCapacity = 5; // default read capacity to 5

    // Set-up max write capacity
    if (dynamoTables[tableDetails.tableName] &&
      dynamoTables[tableDetails.tableName]['readCapacity']) {
      maxCapacity = dynamoTables[tableDetails.tableName]['readCapacity'];
    }

    params.Limit = maxCapacity;

    if (tableDetails.reverseOrder) {
      params.ScanIndexForward = false;
    }

    if (tableDetails.projectionExpression) {
      params.ProjectionExpression = tableDetails.projectionExpression;
    }

    console.log('Get table request: ', JSON.stringify(params));

    let onScan = function(err, data) {
      if (err) {
        console.error(`Unable to get table  ${tableDetails.tableName}. Error: `,
          JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log(`Get table ${tableDetails.tableName} succeeded.  `,
          `${data.Items.length} records returned.`);
        scanDataItems = scanDataItems.concat(data.Items);

        // continue scanning if we have more movies, because
        // scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey !== 'undefined') {
          console.log('Scanning for more data...');
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          client.scan(params, onScan);
        } else {
          resolve(scanDataItems || []);
        }
      }
    };

    client.scan(params, onScan);
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
    conditionExpression: 'attribute_exists(symbol)' (optional expression to
      perform update)
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

    if (updateDetails.conditionExpression) {
      params.ConditionExpression = updateDetails.conditionExpression;
    }

    params.ReturnValues = 'UPDATED_NEW';

    console.log('Update table request: ', JSON.stringify(params));

    client.update(params, function(err, data) {
      if (err && err.code === 'ConditionalCheckFailedException') {
        console.log(`Skipping update to table  ${updateDetails.tableName}.`,
          ` Condition expression not satisifed`);
        resolve({
          result: 'skipped',
          message: `Skipping update to table  ${updateDetails.tableName}.` +
            ' Condition expression not satisifed',
        });
      } else if (err) {
        console.error(`Unable to update table  ${updateDetails.tableName}.` +
          ' Error:', JSON.stringify(err, null, 2));
        reject(JSON.stringify(err, null, 2));
      } else {
        console.log(`Update table ${updateDetails.tableName} succeeded.`);
        resolve(data || null);
      }
    });
  });
};


// Always set-up table info on load
getTableInfo();

module.exports = {
  insertRecord: insertRecord,
  queryTable: queryTable,
  scanTable: scanTable,
  getTable: getTable,
  updateRecord: updateRecord,
};
