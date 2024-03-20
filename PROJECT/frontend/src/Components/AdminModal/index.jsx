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
    setAdminPop,
    email,
    setEmail,
    password,
    SetPassword,
    warningMsg,
    setWarningMsg,
  } = useStoryContext();
  // let handleSignUpClick = () => {
  //   setAdminPop(false);
  //   setModalPop(true);
  // };
  let handleSignUpClick = (flag) => {
    if (flag === true) {
      setClickedSignUp(false);
    } else {
      setClickedSignUp(true);
    }
  };
  let onClickModalPop = () => {
    setAdminPop(false);
    document.getElementsByClassName("home-page")[0].removeAttribute("style");
  };
  let handleSubmit = async () => {
    const res = await axios.get(
      `${process.env.REACT_APP_BACKEND_URL}api/admin-login`,
      {
        params: {
          email: email,
          password: password,
        },
      }
    );
    let status = res.data.status;
    if (status === 400) {
      setWarningMsg("Please Enter valid email-id/password");
    } else if (status === 401) {
      setWarningMsg("Admin credentials not found in database");
    } else if (status === 402) {
      setWarningMsg("Please enter correct password");
    } else if (status === 200) {
      setWarningMsg("Login Successful!");
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("user-email",email);
      navigate("/admin/dashboard");
    }
  };

  return (
    <>
      <div className="bg-white/80 blur-sm fixed top-0 left-0 h-screen w-screen z-30" />
      <div className="bg-white border p-8 rounded-xl shadow flex flex-col items-center gap-2 fixed top-1/2 left-1/2 w-full max-w-md h-fit -translate-x-1/2 -translate-y-1/2 z-50">
        <img
          src="https://static.pbcdn.in/myaccount-cdn/images/login-close-icon.svg"
          alt="close button"
          className="admin-modal-close-btn"
          onClick={onClickModalPop}
        />
        {/* <img src="logo.svg" alt="website-logo" className="admin-website-logo" /> */}
        {/* <img
      src="https://static.pbcdn.in/myaccount-cdn/images/login-illusion.png"
      alt="modal-caption-img"
      className="modal-caption-img"
    /> */}

        <img
          src="https://static.pbcdn.in/myaccount-cdn/images/login-illusion.png"
          alt="modal-caption-img"
          className="admin-modal-caption-img"
        />
        <p className="text-lg my-8">
          {clickedSignUp
            ? `To Sign up, please enter your email-id & password.`
            : `To Login, please enter your registered email-id & password.`}
        </p>

        {/* <p className="admin-modal-caption-text">
      The Admin Login, please enter your email-id & password.
    </p> */}
        <div className="flex flex-col items-center w-80 gap-8">
          <input
            type="email"
            name="email"
            className="p-2 rounded outline-none w-full"
            // placeholder="Enter Admin email-id"
            placeholder={
              clickedSignUp
                ? "Enter Admin email-id"
                : "Enter Admin registered email-id"
            }
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            type="password"
            name="password"
            className="p-2 rounded outline-none w-full"
            // placeholder="Enter Admin password"
            placeholder={
              clickedSignUp
                ? "Enter Admin password"
                : "Enter Admin registered password"
            }
            onChange={(e) => SetPassword(e.target.value)}
            required
          />
          <p className="warning-msg-admin-modal">{warningMsg}</p>
          <button
            type="submit"
            className="p-2 rounded outline-none w-full hover:bg-primary-600 bg-primary-700 text-white"
            onClick={() => handleSubmit(clickedSignUp)}
          >
            {/* Admin Login */}
            {clickedSignUp ? `Sign up` : `Login`}
          </button>

          <p className="first-time-user-admin-modal">
            {/* Are you a normal user? &nbsp; */}
            {clickedSignUp ? `Are you a normal user?` : `First time user?`}{" "}
            <span
              className="sign-up-admin-modal"
              //  onClick={handleSignUpClick}>
              onClick={() => handleSignUpClick(clickedSignUp)}
            >
              {/* Sign up */}
              {clickedSignUp ? `Login` : `Sign up`}
            </span>
          </p>
        </div>
      </div>
    </>
  );
}


// import React from "react";
// import "./style.css";
// import useStoryContext from "../../hooks/useProductContext";
// import axios from "axios";
// import { useNavigate } from "react-router-dom";

// export default function Index() {
//   let navigate = useNavigate();
//   let { clickedSignUp,  setClickedSignUp,setModalPop, setAdminPop, email, setEmail, password, SetPassword,
//     warningMsg,setWarningMsg } =
//     useStoryContext();
//   // let handleSignUpClick = () => {
//   //   setAdminPop(false);
//   //   setModalPop(true);
//   // };
//   let handleSignUpClick = (flag) => {
//     if (flag === true) {
//       setClickedSignUp(false);
//     } else {
//       setClickedSignUp(true);
//     }
//   };
//   let onClickModalPop = () => {
//     setAdminPop(false);
//     document.getElementsByClassName("home-page")[0].removeAttribute("style");
//   };
//   let handleSubmit = async () => {
//     const res = await axios.get(
//       `${process.env.REACT_APP_BACKEND_URL}api/admin-login`,
//       {
//         params: {
//           email: email,
//           password: password,
//         },
//       }
//     );
//     let status = res.data.status;
//     if (status === 400) {
//       setWarningMsg("Please Enter valid email-id/password");
//     }
//     else if (status === 401){
//       setWarningMsg("Admin credentials not found in database");
//     }
//     else if (status === 402){
//       setWarningMsg("Please enter correct password")
//     }

//     else if (status === 200){
//       setWarningMsg("Login Successful!");
//       localStorage.setItem("token",res.data.token);
//     }
    

//   };

//   return (
//     <div className="bg-white border p-8 rounded-xl shadow flex flex-col items-center gap-2 fixed top-1/2 left-1/2 max-w-sm h-fit -translate-x-1/2 -translate-y-1/2">
//       <img
//         src="https://static.pbcdn.in/myaccount-cdn/images/login-close-icon.svg"
//         alt="close button"
//         className="admin-modal-close-btn"
//         onClick={onClickModalPop}
//       />
//       {/* <img src="logo.svg" alt="website-logo" className="admin-website-logo" /> */}
//       {/* <img
//         src="https://static.pbcdn.in/myaccount-cdn/images/login-illusion.png"
//         alt="modal-caption-img"
//         className="modal-caption-img"
//       /> */}
     
//         <img
//           src="https://static.pbcdn.in/myaccount-cdn/images/login-illusion.png"
//           alt="modal-caption-img"
//           className="admin-modal-caption-img"
//         />
//         <p className="modal-caption-text">
//         {clickedSignUp
//           ? `To Sign up, please enter your email-id & password.`
//           : `To Login, please enter your registered email-id & password.`}
//       </p>

//         {/* <p className="admin-modal-caption-text">
//           The Admin Login, please enter your email-id & password.
//         </p> */}
//       <div className="flex flex-col items-center w-80 gap-8">
//         <input
//           type="email"
//           name="email"
//           className="p-2 rounded outline-none w-full"
//           // placeholder="Enter Admin email-id"
//           placeholder={
//             clickedSignUp
//               ? "Enter Admin email-id"
//               : "Enter Admin registered email-id"
//           }
//           onChange={(e) => setEmail(e.target.value)}
//           required
//         />
//         <input
//           type="password"
//           name="password"
//           className="p-2 rounded outline-none w-full"
//           // placeholder="Enter Admin password"
//           placeholder={
//             clickedSignUp
//               ? "Enter Admin password"
//               : "Enter Admin registered password"
//           }
//           onChange={(e) => SetPassword(e.target.value)}
//           required
//         />
//         <p className="warning-msg-admin-modal">{warningMsg}</p>
//         <button
//           type="submit"
//           className="p-2 rounded outline-none w-full hover:bg-primary-600 bg-primary-700 text-white"
//           onClick={() => handleSubmit(clickedSignUp)}
//         >
//           {/* Admin Login */}
//           {clickedSignUp ? `Sign up` : `Login`}
//         </button>

//         <p className="first-time-user-admin-modal">
//           {/* Are you a normal user? &nbsp; */}
//           {clickedSignUp ? `Are you a normal user?` : `First time user?`}{" "}
//           <span className="sign-up-admin-modal"
//           //  onClick={handleSignUpClick}>
//           onClick={() => handleSignUpClick(clickedSignUp)}
//           >
//             {/* Sign up */}
//             {clickedSignUp ? `Login` : `Sign up`}
//           </span>
//         </p>
//       </div>
//     </div>
//   );
// }
