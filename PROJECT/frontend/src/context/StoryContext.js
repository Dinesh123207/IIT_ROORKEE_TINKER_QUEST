const { createContext, useState } = require("react");

const StoryContext = createContext();

const Provider = ({ children }) => {
  const [loggedIn, setLoggedIn] = useState(false);
  const [clickedSignUp,setClickedSignUp] = useState(false);
  const [modalPop,setModalPop] = useState(false);
  const [adminPop,setAdminPop] = useState(false);
  const [email,setEmail] = useState("");
  const [password,SetPassword] = useState("");
  const [warningMsg,setWarningMsg] = useState(null);

  const valueToShare = {
    loggedIn,
    setLoggedIn,
    clickedSignUp,
    setClickedSignUp,
    modalPop,
    setModalPop,
    adminPop,
    setAdminPop,
    email,setEmail,
    password,SetPassword,
    warningMsg,setWarningMsg,
  };

  return (
    <StoryContext.Provider value={valueToShare}>
      {children}
    </StoryContext.Provider>
  );
};

export { Provider };

export default StoryContext;
