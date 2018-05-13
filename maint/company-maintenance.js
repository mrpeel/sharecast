'use strict';

const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const symbols = require('../retrieve-data/dynamo-symbols');
const dynamodb = require('../retrieve-data/dynamodb');
const maintRecords = require('./maint-symbols.json');

let cleanupOldSymbols = asyncify(function () {
  // Prep local access
  dynamodb.setLocalAccessConfig();
  // Execute delete
  maintRecords.symbolsToRemove.forEach((symbol) => {
    // Execute delete
    awaitify(symbols.removeCompany(symbol));
  });
});

let addNewCompanies = asyncify(function () {
  // Prep local access
  dynamodb.setLocalAccessConfig();
  // Execute adds
  maintRecords.symbolsToAdd.forEach((symbolRecord) => {
    // Execute delete
    awaitify(symbols.addCompany(symbolRecord));
  });
});


// cleanupOldSymbols();
addNewCompanies();