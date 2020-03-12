'use strict';

const asyncify = require('asyncawait/async');
const awaitify = require('asyncawait/await');
const symbols = require('../retrieve-data/dynamo-symbols');
const dynamodb = require('../retrieve-data/dynamodb');
const maintRecords = require('./maint-symbols.json');

let cleanupOldSymbols = asyncfunction () {
  // Prep local access
  dynamodb.setLocalAccessConfig();
  // Execute delete
  maintRecords.symbolsToRemove.forEach((symbol) => {
    // Execute delete
    awaitsymbols.removeCompany(symbol));
  });
});

let addNewCompanies = asyncfunction () {
  // Prep local access
  dynamodb.setLocalAccessConfig();
  // Execute adds
  maintRecords.symbolsToAdd.forEach((symbolRecord) => {
    // Execute delete
    awaitsymbols.addCompany(symbolRecord));
  });
});


// cleanupOldSymbols();
addNewCompanies();