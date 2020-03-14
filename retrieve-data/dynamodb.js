'use strict';

const AWS = require('aws-sdk');
const moment = require('moment-timezone');
let maxExecutionTime = 900;

// Set the base retry to wait 500ms
AWS.config.update({
  retryDelayOptions: {
    base: 500,
  },
  maxRetries: 50,
});

let client = new AWS.DynamoDB.DocumentClient();
let dynamoTables = {};


/** Sets access to AWS for local execution
 */
let setLocalAccessConfig = function() {
  AWS.config.loadFromPath(`${__dirname}/../credentials/aws.json`);
  client = new AWS.DynamoDB.DocumentClient();
  /* Set max execution time to a large number to prevent
      re-invocation of lambda when running locally */
  maxExecutionTime = 1000000;
};

let getExecutionMaxTime = function() {
  return maxExecutionTime;
};

/** Gets all the dynamo tables and records their provisioned capacity
 * @return {Boolean} whether the get info worked
 */
let getTableInfo = async function() {
  // Check if already done`
  if (!Object.keys(dynamoTables).length) {
    let db = new AWS.DynamoDB();

    let tableNames = await listTables(db);

    for (const tableName of tableNames) {
      try {
        let tableData = await describeTable(db, tableName);
        dynamoTables[tableName] = {};
        dynamoTables[tableName]['readCapacity'] = tableData.ProvisionedThroughput.ReadCapacityUnits;
        dynamoTables[tableName]['writeCapacity'] = tableData.ProvisionedThroughput.WriteCapacityUnits;
      } catch (err) {
        console.error(err);
        return false;
      }
    };
    return true;
  } else {
    return true;
  }
};

/** Returns tab;le info from dynamo
 * @param {Object} db - dynamo db reference
 * @param {String} tableName - name of table to describe
 * @return {Promise} promise with table info
 */
let describeTable = async function(db, tableName) {
  try {
    let data = await db.describeTable({
      TableName: tableName,
    }).promise();
    return data.Table;
  } catch (err) {
    throw err;
  }
};

/** Returns a list of dynamo db tables
 * @param {Object} db - dynamo db reference
 * @return {Array} table names
 */
let listTables = async function(db) {
  try {
    let data = await db.listTables({}).promise();
    return data.TableNames;
  } catch (err) {
    throw err;
  }
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
let insertRecord = async function(insertDetails) {
  await getTableInfo();

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
    ReturnConsumedCapacity: 'TOTAL',
  };

  if (insertDetails.primaryKey) {
    params.ConditionExpression = 'attribute_not_exists(' +
      insertDetails.primaryKey[0] + ')';
  }


  // Retrieve primary key vals
  let keyVals = [];
  insertDetails.primaryKey.forEach((keyValue) => {
    keyVals.push(insertDetails.values[keyValue]);
  });

  console.log('Put table request: ', insertDetails.tableName,
    ' ', JSON.stringify(keyVals));

  try {
    await client.put(params).promise();
    console.log(`PutItem to ${insertDetails.tableName} succeeded.`);
    return {
      result: 'inserted',
    };
  } catch (err) {
    if (err.code === 'ConditionalCheckFailedException') {
      console.log(`Skipping add to ${insertDetails.tableName} : `,
        JSON.stringify(insertDetails.primaryKey), ' already exists');
      return {
        result: 'skipped',
        message: `Skipping add to ${insertDetails.tableName} : ` +
          JSON.stringify(insertDetails.primaryKey) + ' already exists.',
      };
    } else {
      console.error(`Unable to add to ${insertDetails.tableName} values: `,
        '. Error JSON:', JSON.stringify(err, null, 2));
      throw new Error({
        result: 'failed',
        message: JSON.stringify(err, null, 2),
      });
    }
  }
};

/** Query a table and return any matching records
* @param {Object} queryDetails - an object with all the details for insert
* queryDetails = {
*  tableName: 'companies',
*  indexName: 'company-created-index' (optional secondary index to use),
*  keyConditionExpression: '#symbol = :symbol',
*  expressionAttributeValues: {
*  ':symbol': 'AAD',
*  },
*  expressionAttributeNames: {
*  '#symbol': 'symbol',
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
let queryTable = async function(queryDetails) {
  await getTableInfo();

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

  if (queryDetails.expressionAttributeNames) {
    params.ExpressionAttributeNames = queryDetails.expressionAttributeNames;
  }

  console.log('Query table request: ', JSON.stringify(params));

  let continueQuerying = true;

  while (continueQuerying) {
    try {
      let data = await client.query(params).promise();
      queryDataItems = queryDataItems.concat(data.Items);

      // continue querying if we have more records, because
      // query can retrieve a maximum of 1MB of data
      if (typeof data.LastEvaluatedKey !== 'undefined' &&
        (!resultsLimit || queryDataItems.length < resultsLimit)) {
        // console.log('Querying for more data...');
        params.ExclusiveStartKey = data.LastEvaluatedKey;
        continueQuerying = true;
      } else {
        console.log(`Querying returned ${queryDataItems.length} items`);
        continueQuerying = false;
        return (queryDataItems || []);
      }
    } catch (err) {
      console.error(`Unable to query ${queryDetails.tableName}. Error: `,
        JSON.stringify(err, null, 2));
      throw new Error(JSON.stringify(err, null, 2));
    }
  }
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
let scanTable = async function(scanDetails) {
  await getTableInfo();

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

  let continueScanning = true;

  while (continueScanning) {
    try {
      let data = await client.scan(params).promise();

      scanDataItems = scanDataItems.concat(data.Items);

      // continue scanning if we have more reords, because
      // scan can retrieve a maximum of 1MB of data
      if (typeof data.LastEvaluatedKey !== 'undefined' &&
        (!resultsLimit || scanDataItems.length < resultsLimit)) {
        // console.log('Scanning for more data...');
        params.ExclusiveStartKey = data.LastEvaluatedKey;
        continueScanning = true;
      } else {
        console.log(`Scan returned ${scanDataItems.length} items`);
        continueScanning = false;
        return (scanDataItems || []);
      }
    } catch (err) {
      console.error(`Unable to scan ${scanDetails.tableName}. Error: `,
        JSON.stringify(err, null, 2));
      throw new Error(JSON.stringify(err, null, 2));
    }
  }
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
let getTable = async function(tableDetails) {
  await getTableInfo();

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

  let continueScanning = true;

  while (continueScanning) {
    try {
      let data = await client.scan(params).promise();
      scanDataItems = scanDataItems.concat(data.Items);

      // continue scanning if we have more movies, because
      // scan can retrieve a maximum of 1MB of data
      if (typeof data.LastEvaluatedKey !== 'undefined') {
        // console.log('Scanning for more data...');
        params.ExclusiveStartKey = data.LastEvaluatedKey;
        continueScanning = true;
      } else {
        console.log(`Get table returned ${scanDataItems.length} items`);
        continueScanning = false;
        return (scanDataItems || []);
      }
    } catch (err) {
      console.error(`Unable to get table  ${tableDetails.tableName}. Error: `,
        JSON.stringify(err, null, 2));
      throw new Error(JSON.stringify(err, null, 2));
    }
  }
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
let updateRecord = async function(updateDetails) {
  await getTableInfo();

  let params = {
    TableName: updateDetails.tableName,
    Key: updateDetails.key,
    UpdateExpression: updateDetails.updateExpression,
    ExpressionAttributeValues: updateDetails.expressionAttributeValues,
  };

  // Make sure updates have a timestamp
  params.ExpressionAttributeValues[':updated'] = moment().tz('Australia/Sydney').format();
  params.UpdateExpression = params.UpdateExpression +
    ', #updated = :updated';

  if (updateDetails.expressionAttributeNames) {
    params.ExpressionAttributeNames = updateDetails.expressionAttributeNames;
    params.ExpressionAttributeNames['#updated'] = 'updated';
  } else {
    params.ExpressionAttributeNames = {
      '#updated': 'updated',
    };
  }

  if (updateDetails.conditionExpression) {
    params.ConditionExpression = updateDetails.conditionExpression;
  }

  params.ReturnValues = 'UPDATED_NEW';

  console.log('Update table request: ', updateDetails.tableName,
    ' ', JSON.stringify(params.Key));

  try {
    let data = await client.update(params).promise();
    console.log(`Update table ${updateDetails.tableName} succeeded.`);
    return (data || null);
  } catch (err) {
    if (err.code === 'ConditionalCheckFailedException') {
      console.log(`Skipping update to table  ${updateDetails.tableName}.`,
        ` Condition expression not satisifed`);
      return {
        result: 'skipped',
        message: `Skipping update to table  ${updateDetails.tableName}.` +
          ' Condition expression not satisifed',
      };
    } else {
      console.error(`Unable to update table  ${updateDetails.tableName}.` +
        ' Error:', JSON.stringify(err, null, 2));
      throw new Error(JSON.stringify(err, null, 2));
    }
  }
};


/** Delets a record from dynamodb
* @param {Object} deleteDetails - an object with all the details for insert
* deleteDetails = {
*  tableName: 'companies',
*  key: {
    symbol: 'JBH'
  },
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
let deleteRecord = async function(deleteDetails) {
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

  try {
    await client.delete(params).promise();
    console.log(`Delete item from ${deleteDetails.tableName} succeeded.`);
    return {
      result: 'deleted',
    };
  } catch (err) {
    console.error(`Unable to delete ${deleteDetails.key}`,
      '. Error JSON:', JSON.stringify(err, null, 2));
    throw new Error({
      result: 'failed',
      message: JSON.stringify(err, null, 2),
    });
  }
};

let sleep = function(ms) {
  if (!ms) {
    ms = 1;
  }
  return new Promise((r) => {
    setTimeout(r, ms);
  });
};

let getRandomInt = function(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  // The maximum is exclusive and the minimum is inclusive
  return Math.floor(Math.random() * (max - min)) + min;
};

module.exports = {
  insertRecord: insertRecord,
  queryTable: queryTable,
  scanTable: scanTable,
  getTable: getTable,
  updateRecord: updateRecord,
  deleteRecord: deleteRecord,
  setLocalAccessConfig: setLocalAccessConfig,
  getExecutionMaxTime: getExecutionMaxTime,
};
