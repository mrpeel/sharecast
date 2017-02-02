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
  queryDb: function(connection, query) {
    return new Promise(function(resolve, reject) {
      if (!connection || !query) {
        reject('Missing parameter, connection: ' + connection || '' +
          ', query: ' + query || '');
        return;
      }

      connection.query(query)
        .then((rows) => {
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
