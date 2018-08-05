module.exports = {
    "extends": "eslint-config-google",
    "parserOptions": {
        "ecmaVersion": 2017
    },
    "rules": {
        "max-len": ["error", {
            "code": 120,
            "tabWidth": 2,
            "ignoreStrings": true,
            "ignoreComments": true,
        }]
    }
};