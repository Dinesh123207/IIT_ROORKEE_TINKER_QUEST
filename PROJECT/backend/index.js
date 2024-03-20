const express = require("express");
const app = express();
const dotEnv = require("dotenv");
const bodyParser = require("body-parser");
const mongoose = require("mongoose");
const cors = require("cors");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const User = require("./models/user");
const Admin = require("./models/admin");

dotEnv.config();

app.use(cors());
app.use(bodyParser.urlencoded({ extended: false }));

let isValidNumber = (num) => {
  let len = Math.ceil(Math.log10(num + 1)) - 1;

  if (len === 10) return true;
  else return false;
};

let isValidEmail = (email) => {
  let re =
    /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/; // eslint-disable-line
  if (re.test(email)) {
    return true;
  }

  return false;
};
const IsFilledForm = (req, res, next) => {
  let { email, password } = req.query;

  if (
    !email ||
    !password ||
    !isValidEmail(email) ||
    password.trim().length === 0
  ) {
    res.json({ status: 400, message: "Invalid email/pwd" });
  } else {
    next();
  }
};

const isUserExist = async (req, res, next) => {
  const { email } = req.query;
  const user = await User.findOne({ email: email });

  if (user) {
    res.json({ status: 403, message: "User already exist, please login" });
  } else {
    next();
  }
};

const isUserRegistered = async (req, res, next) => {
  const { email } = req.query;
  const user = await User.findOne({ email: email });

  if (user) {
    next();
  } else {
    res.json({ status: 405, message: "User not signed up" });
  }
};

const isAdminRegistered = async (req, res, next) => {
  const { email, password } = req.query;

  if (
    !email ||
    !password ||
    !isValidEmail(email) ||
    password.trim().length === 0
  ) {
    res.json({ status: 400, message: "Invalid email/pwd" });
  } else {
    const admin = await Admin.findOne({ email: email });

    if (admin) {
      next();
    } else {
      res.json({ status: 401, message: "Admin not found in database" });
    }
  }
};

app.get("/", (req, res) => {
  res.json({ message: "All operational!" });
});

app.get("/api/signup", IsFilledForm, isUserExist, async (req, res) => {
  let { email, password } = req.query;
  const encryptedPassword = await bcrypt.hash(password, 10);
  let token = jwt.sign({ email, password }, process.env.JWT_SECRET);
  await new User({
    email: email,
    password: encryptedPassword,
    token: token,
  }).save();
  res.json({ status: 201, message: "User created!", token: token });
});

app.get("/api/login", IsFilledForm, isUserRegistered, async (req, res) => {
  let { email, password } = req.query;
  await User.findOne({ email })
    .then(async (user) => {
      let isMatchedPwd = await bcrypt.compare(password, user.password);
      if (isMatchedPwd) {
        let token = jwt.sign({ email, password }, process.env.JWT_SECRET);
        await User.findOneAndUpdate({ email: user.email }, { token: token });
        res.json({ status: 202, message: "User Logged in", token: token });
      } else {
        res.json({ status: 402, mesage: "Please enter correct password" });
      }
    })
    .catch((err) => {
      console.log(err);
    });
});

app.get("/api/admin-login", isAdminRegistered, (req, res) => {
  const { email, password } = req.query;
  Admin.findOne({ email })
    .then(async (admin) => {
      if (admin.password === password) {
        let token = jwt.sign({ email, password }, process.env.JWT_SECRET);
        await Admin.findOneAndUpdate({ email: admin.email }, { token: token });
        res.json({ status: 200, message: "Admin Logged in", token: token });
      } else {
        res.json({ status: 402, mesage: "Please enter correct password" });
      }
    })
    .catch((err) => {
      console.log(err);
    });
});

app.get("/api/get-user-claim-detail", (req, res) => {
  const { email, token, name, mobile } = req.query;
  let decoded = jwt.verify(token, process.env.JWT_SECRET);
  if (decoded.email === email) {
    User.findOne({ email })
      .then(async (user) => {
        await User.findOneAndUpdate(
          { email: user.email },
          { name: name, mobile: mobile }
        );
        res.json({ status: 200, message: "User details updated in database" });
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    return res.json({ status: 401, message: "Unauthorized access!" });
  }
});

app.get("/api/claim-the-document", (req, res) => {
  let { email, token, document_name, client_id, claim_status } = req.query;
  let decoded = jwt.verify(token, process.env.JWT_SECRET);
  if (decoded.email === email) {
    User.findOne({ email })
      .then(async (user) => {
        document_name = [...user.document_name, document_name];
        client_id = [...user.client_id, client_id];
        claim_status = [...user.claim_status, claim_status];
        await User.findOneAndUpdate(
          { email: user.email },
          {
            document_name: document_name,
            client_id: client_id,
            claim_status: claim_status,
          }
        );
        res.json({ status: 200, message: "User Claimed!" });
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    return res.json({ status: 401, message: "Unauthorized access!" });
  }
});

app.get("/api/get-admin-documents/:docs_status", async (req, res) => {
  const { docs_status } = req.params;
  const { token, email } = req.query;
  let decoded = jwt.verify(token, process.env.JWT_SECRET);
  if (decoded.email === email) {
    let docs = await User.find({});
    let result = [];
    for (let i = 0; i < docs.length; i++) {
      let doc = docs[i];
      for (let j = 0; j < doc.claim_status.length; j++) {
        let temp = {};
        temp.name = doc.name;
        temp.email = doc.email;
        temp.mobile = doc.mobile;
        if (doc.claim_status[j] === docs_status) {
          temp.claim_status = doc.claim_status[j];
          temp.client_id = doc.client_id[j];
          temp.document_name = doc.document_name[j];
          result.push(temp);
        }
      }
    }
    res.json({ result });
  } else {
    res.json({ status: 401, mesage: "Not authorized" });
  }
});

app.get("/api/change-admin-document-status", async (req, res) => {
  const { token, admin_email, feedback, curr_status, client_id, user_email } =
    req.query;
  let decoded = jwt.verify(token, process.env.JWT_SECRET);
  if (decoded.email === admin_email) {
    await User.findOne({ email: user_email }).then(async (user) => {
      for (let i = 0; i < user.client_id.length; i++) {
        if (user.client_id[i] === client_id) {
          let temp_arr = [...user.feedback];
          temp_arr[i] = feedback;
          let temp_arr_2 = [...user.claim_status];
          temp_arr_2[i] = curr_status;
          await User.findOneAndUpdate(
            { email: user_email },
            {
              claim_status: temp_arr_2,
              feedback: temp_arr,
            }
          );
          break;
        }
      }
    });
    res.json({ status: 200, message: "Document status changed!" });
  } else {
    res.json({ status: 401, mesage: "Not authorized" });
  }
});

app.get("/api/user-claim-history", async (req, res) => {
  const { token, email } = req.query;
  let decoded = jwt.verify(token, process.env.JWT_SECRET);
  if (decoded.email === email) {
    await User.findOne({ email })
      .then((user) => {
        let history = [];
        for (let i = 0; i < user.claim_status.length; i++) {
          let temp_obj = {};
          temp_obj.name = user.name;
          temp_obj.email = user.email;
          temp_obj.mobile = user.mobile;
          temp_obj.client_id = user.client_id[i];
          temp_obj.document_name = user.document_name[i];
          temp_obj.feedback = user.feedback[i];
          temp_obj.claim_status = user.claim_status[i];
          history.push(temp_obj);
        }

        res.json({ history });
        // res.json({user});
      })
      .catch((err) => {
        console.log(err);
      });
  } else {
    res.json({ status: 401, mesage: "Not authorized", decoded });
  }
});

app.listen(process.env.SERVER_PORT, () => {
  mongoose
    .connect(process.env.MONGODB_URL, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    })
    .then(() => {
      console.log("Connected to MongoDB");
      console.log(`Server running on port ${process.env.SERVER_PORT}`);
    })
    .catch(() => {
      console.log("Could not connect to MongoDB");
    });
});
