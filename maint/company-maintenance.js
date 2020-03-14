'use strict';

const symbols = require('../retrieve-data/dynamo-symbols');
const dynamodb = require('../retrieve-data/dynamodb');
const maintRecords = require('./maint-symbols.json');

let cleanupOldSymbols = async function () {
  // Prep local access
  dynamodb.setLocalAccessConfig();
  // Execute delete
  maintRecords.symbolsToRemove.forEach((symbol) => {
    // Execute delete
    await symbols.removeCompany(symbol);
  });
};

let addNewCompanies = async function () {
  // Prep local access
  dynamodb.setLocalAccessConfig();
  // Execute adds
  maintRecords.symbolsToAdd.forEach((symbolRecord) => {
    // Execute delete
    await symbols.addCompany(symbolRecord);
  });
};


// cleanupOldSymbols();
addNewCompanies();