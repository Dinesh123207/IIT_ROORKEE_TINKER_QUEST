const mongoose = require('mongoose');

const Admin = mongoose.model("Admin",{
    email: String,
    password: String,
    mobile: Number,
    name: String,
    token: String
})

module.exports = Admin;