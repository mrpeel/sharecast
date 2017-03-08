'use strict';

const AWS = require('aws-sdk');
const moment = require('moment-timezone');
const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');


AWS.config.loadFromPath('../credentials/aws.json');

const client = new AWS.DynamoDB.DocumentClient();
let dynamoTables = {};

/** Gets all the dynamo tables and records their provisioned capacity
* @return {Boolean} whether the get info worked
*/
let getTableInfo = asyncify(function() {
  // Check if already done`
  if (!Object.keys(dynamoTables).length) {
    let db = new AWS.DynamoDB();

    let tableNames = awaitify(listTables(db));

    tableNames.forEach((tableName) => {
      try {
        let tableData = awaitify(describeTable(db, tableName));
        dynamoTables[tableName] = {};
        dynamoTables[tableName]['readCapacity'] = tableData.ProvisionedThroughput.ReadCapacityUnits;
        dynamoTables[tableName]['writeCapacity'] = tableData.ProvisionedThroughput.WriteCapacityUnits;
      } catch (err) {
        console.error(err);
        return false;
      }
    });
    return true;
  } else {
    return true;
  }
});

/** Returns tab;le info from dynamo
* @param {Object} db - dynamo db reference
* @param {String} tableName - name of table to describe
* @return {Promise} promise with table info
*/
let describeTable = function(db, tableName) {
  return new Promise(function(resolve, reject) {
    db.describeTable({
      TableName: tableName,
    }, function(err, data) {
      if (err) {
        reject(err);
      } else {
        resolve(data.Table);
      }
    });
  });
};

/** Returns a list of dynamo db tables
* @param {Object} db - dynamo db reference
* @return {Array} table names
*/
let listTables = function(db) {
  return new Promise(function(resolve, reject) {
    db.listTables({}, function(err, data) {
      if (err) {
        reject(err);
      } else {
        resolve(data.TableNames);
      }
    });
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

    awaitify(getTableInfo());


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


    console.log('Put table request: ', insertDetails.tableName);

    client.put(params, function(err, data) {
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

    awaitify(getTableInfo());

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
        /* console.log(`Query table ${queryDetails.tableName} succeeded.  `,
          `${data.Items.length} records returned.`);*/

        queryDataItems = queryDataItems.concat(data.Items);

        // continue scanning if we have more movies, because
        // scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey !== 'undefined' &&
          (!resultsLimit || queryDataItems.length < resultsLimit)) {
          // console.log('Querying for more data...');
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          client.query(params, onQuery);
        } else {
          console.log(`Querying returned ${queryDataItems.length} items`);
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

    awaitify(getTableInfo());

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
        /* console.log(`Scan table ${scanDetails.tableName} succeeded. `,
          `${data.Items.length} records returned.`); */

        scanDataItems = scanDataItems.concat(data.Items);

        // continue scanning if we have more movies, because
        // scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey !== 'undefined' &&
          (!resultsLimit || scanDataItems.length < resultsLimit)) {
          // console.log('Scanning for more data...');
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          client.scan(params, onScan);
        } else {
          console.log(`Scan returned ${scanDataItems.length} items`);
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
let getTable = asyncify(function(tableDetails) {
  return new Promise(function(resolve, reject) {
    let params = {
      TableName: tableDetails.tableName,
    };

    let scanDataItems = [];

    let maxCapacity = 5; // default read capacity to 5

    awaitify(getTableInfo());

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
        /* console.log(`Get table ${tableDetails.tableName} succeeded.  `,
          `${data.Items.length} records returned.`); */
        scanDataItems = scanDataItems.concat(data.Items);

        // continue scanning if we have more movies, because
        // scan can retrieve a maximum of 1MB of data
        if (typeof data.LastEvaluatedKey !== 'undefined') {
          // console.log('Scanning for more data...');
          params.ExclusiveStartKey = data.LastEvaluatedKey;
          client.scan(params, onScan);
        } else {
          console.log(`Get table returned ${scanDataItems.length} items`);
          resolve(scanDataItems || []);
        }
      }
    };

    client.scan(params, onScan);
  });
});

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

    awaitify(getTableInfo());

    if (updateDetails.expressionAttributeNames) {
      params.ExpressionAttributeNames = updateDetails.expressionAttributeNames;
    }

    if (updateDetails.conditionExpression) {
      params.ConditionExpression = updateDetails.conditionExpression;
    }

    params.ReturnValues = 'UPDATED_NEW';

    console.log('Update table request: ', updateDetails.tableName,
      ' ', JSON.stringify(params.key));

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


/** Delets a record from dynamodb
* @param {Object} deleteDetails - an object with all the details for insert
* insertDetails = {
*  tableName: 'companies',
*  key: [
*    'companySymbol',
*  ],
*  conditionExpression: 'attribute_exists(symbol)' (optional expression to
  perform delete)
  expressionAttributeValues: {
     ':sentTime': moment().tz('Australia/Sydney').format(),
     ':sentLog': result
   },
    expressionAttributeNames: (optional) {
       '#sentTime': 'SentTime',
    };
@return {Promise} which resolves with:
* {
*   result: deleted / failed
*   message: error message (optional)
* }
*/
let deleteRecord = function(deleteDetails) {
  return new Promise(function(resolve, reject) {
    // Set-up item details for insert
    let params = {
      TableName: deleteDetails.tableName,
      Key: deleteDetails.key,
    };

    if (deleteDetails.expressionAttributeValues) {
      params.ExpressionAttributeValues = deleteDetails.expressionAttributeValues;
    }

    if (deleteDetails.expressionAttributeNames) {
      params.ExpressionAttributeNames = deleteDetails.expressionAttributeNames;
    }


    if (deleteDetails.conditionExpression) {
      params.ConditionExpression = deleteDetails.conditionExpression;
    }


    console.log('Delete item request: ', JSON.stringify(params));

    client.delete(params, function(err, data) {
      if (err) {
        console.error(`Unable to delete ${deleteDetails.key}`,
          '. Error JSON:', JSON.stringify(err, null, 2));
        reject({
          result: 'failed',
          message: JSON.stringify(err, null, 2),
        });
      } else {
        console.log(`Delete item from ${deleteDetails.tableName} succeeded.`);
        resolve({
          result: 'deleted',
        });
      }
    });
  });
};


module.exports = {
  insertRecord: insertRecord,
  queryTable: queryTable,
  scanTable: scanTable,
  getTable: getTable,
  updateRecord: updateRecord,
  deleteRecord: deleteRecord,
};
