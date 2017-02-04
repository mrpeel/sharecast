const mysql = require('promise-mysql');

module.exports = {
  connectToDb: function(host, user, password, db) {
    return new Promise(function(resolve, reject) {
      if (!host || !user || !password || !db) {
        reject('Missing parameter, host: ' + host || '' +
          ', user: ' + user || '' + ', password: ' + password || '' +
          ', db: ' + db || '');
        return;
      }

      mysql.createConnection({
        host: host,
        user: user,
        password: password,
        database: db,
      })
        .then((conn) => {
          resolve(conn);
        })
        .catch((err) => {
          reject(err);
        });
    });
  },
  selectQuery: function(connection, query) {
    return new Promise(function(resolve, reject) {
      if (!connection || !query) {
        reject('Missing parameter, connection: ' + connection || '' +
          ', query: ' + query || '');
        return;
      }

      if (query.indexOf('SELECT') !== 0) {
        reject('selectQuery call does not start with \'SELECT\'');
        return;
      }

      console.log('Select query: ' + query);

      connection.query(query)
        .then((rows) => {
          console.log(rows.length + ' rows returned');
          resolve(rows);
        })
        .catch((err) => {
          reject(err);
        });
    });
  },
  executeQuery: function(connection, query) {
    return new Promise(function(resolve, reject) {
      if (!connection || !query) {
        reject('Missing parameter, connection: ' + connection || '' +
          ', query: ' + query || '');
      // return;
      }

      if (query.indexOf('UPDATE') !== 0 && query.indexOf('INSERT') !== 0 &&
        query.indexOf('DELETE') !== 0) {
        reject('executeQuery call does not start with one of: \'INSERT\', ' +
          '\'UPDATE\', \'DELETE\'');
        return;
      }
      console.log('Execute query: ' + query);

      connection.query(query)
        .then((rows) => {
          if (rows.affectedRows !== undefined) {
            console.log(rows.affectedRows + ' rows affected');
          } else {
            console.log(rows);
          }
          resolve(rows);
        })
        .catch((err) => {
          reject(err);
        });
    });
  },
  closeConnection: function(connection) {
    try {
      connection.end();
    } catch (err) {
      console.log(err);
    }
  },
};
