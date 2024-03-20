import React from "react";
import "./style.css";
import useStoryContext from "../../hooks/useProductContext";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function Index() {
  let navigate = useNavigate();
  let {
    clickedSignUp,
    setClickedSignUp,
    setModalPop,
    warningMsg,
    setWarningMsg,
    email,
    setEmail,
    password,
    SetPassword,
  } = useStoryContext();
  let handleSignUpClick = (flag) => {
    if (flag === true) {
      setClickedSignUp(false);
    } else {
      setClickedSignUp(true);
    }
  };
  let onClickModalPop = () => {
    setModalPop(false);
    document.getElementsByClassName("home-page")[0].removeAttribute("style");
  };

  let handleSubmit = async (flag) => {
    console.log(flag);
    let res;
    if (flag === true) {
      res = await axios.get(`${process.env.REACT_APP_BACKEND_URL}api/signup`, {
        params: {
          email: email,
          password: password,
        },
      });
    } else {
      res = await axios.get(`${process.env.REACT_APP_BACKEND_URL}api/login`, {
        params: {
          email: email,
          password: password,
        },
      });
    }

    let status = res.data.status;
    console.log(status);
    if (status === 400) {
      setWarningMsg("Please Enter valid email-id/password!");
    } else if (status === 401) {
      setWarningMsg("Admin credentials not found in database!");
    } else if (status === 402) {
      setWarningMsg("Please enter correct password");
    } else if (status === 403) {
      setWarningMsg("User already exist, Please login!");
    } else if (status === 405) {
      setWarningMsg("User not signed up, Please sign up!");
    } else if (status === 201) {
      setWarningMsg("User signed up successfully!");
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("user-email",email);
      navigate("/user/dashboard");
    } else if (status === 202) {
      setWarningMsg("User logged in successfully!");
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("user-email",email);
      navigate("/user/dashboard");
    }
  };

  return (
    <>
    {/* <div className="bg-white/80 blur-sm fixed top-0 left-0 h-screen w-screen z-30"/>
    <div className="bg-white border p-8 rounded-xl shadow flex flex-col items-center gap-2 fixed top-1/2 left-1/2 max-w-sm h-fit -translate-x-1/2 -translate-y-1/2"> */}
      <div className="bg-white/80 blur-sm fixed top-0 left-0 h-screen w-screen z-30" />
      <div className="bg-white border p-8 rounded-xl shadow flex flex-col items-center gap-2 fixed top-1/2 left-1/2 w-full max-w-md h-fit -translate-x-1/2 -translate-y-1/2 z-50">

      <img
        src="https://static.pbcdn.in/myaccount-cdn/images/login-close-icon.svg"
        alt="close button"
        className="modal-close-btn"
        onClick={onClickModalPop}
      />

      <img
        src="https://static.pbcdn.in/myaccount-cdn/images/login-illusion.png"
        alt="modal-caption-img"
        className="modal-caption-img"
      />
      <p className="text-lg my-8">
        {clickedSignUp
          ? `To Sign up, please enter your email-id & password.`
          : `To Login, please enter your registered email-id & password.`}
      </p>
      <div className="flex flex-col items-center w-80 gap-8">
        <input
          type="email"
          name="email"
          className="p-2 rounded outline-none w-full"
          placeholder={
            clickedSignUp
              ? "Enter your email-id"
              : "Enter your registered email-id"
          }
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          name="password"
          className="p-2 rounded outline-none w-full"
          placeholder={
            clickedSignUp
              ? "Enter your password"
              : "Enter your registered password"
          }
          onChange={(e) => SetPassword(e.target.value)}
          required
        />
        {warningMsg ? (
          <p className="warning-msg-admin-modal">{warningMsg}</p>
        ) : null}
        <button
          type="submit"
          className="p-2 rounded outline-none w-full hover:bg-primary-600 bg-primary-700 text-white"
          onClick={() => handleSubmit(clickedSignUp)}
        >
          {clickedSignUp ? `Sign up` : `Login`}
        </button>

        <p className="first-time-user-modal">
          {clickedSignUp ? `Already Signed up?` : `First time user?`}{" "}
          <span
            className="sign-up-modal"
            onClick={() => handleSignUpClick(clickedSignUp)}
          >
            {clickedSignUp ? `Login` : `Sign up`}
          </span>
        </p>
      </div>
    </div>
    </>
  );
}
