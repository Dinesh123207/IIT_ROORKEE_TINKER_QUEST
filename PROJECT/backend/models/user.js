const mongoose = require('mongoose');

const User = mongoose.model("User",{
    email: String,
    password: String,
    mobile: Number,
    name: String,
    token: String,
    document_name: Array,
    claim_status: Array,
    client_id: Array,
    feedback: Array
})

module.exports = User;