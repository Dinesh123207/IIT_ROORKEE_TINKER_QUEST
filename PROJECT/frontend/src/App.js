

import "./App.css";
import { Route, Routes } from "react-router-dom";
import { Provider } from "./context/StoryContext";
import HomePage from "./Components/HomePage";
import Loader from "./Components/Loader";
import UserDashboard from "./Components/UserDashboard";
import AdminDashboard from "./Components/AdminDashboard";
import FilterPage from "./Components/FilterPage";
import ResultsLoader from "./Components/ResultsLoader";
import ResultsPage from "./Components/ResultsPage";
import MultipleFilesUpload from "./Components/MultipleFilesUpload";

function App() {
  return (
    <Provider>
      <div className="App dark:bg-gray-900">
        <Routes>
          <Route path="/loader" element={<Loader/>}/>
          <Route path="/" element={<HomePage />} />
          <Route path="/user/dashboard" element={<UserDashboard/>}/>
          <Route path="/admin/dashboard" element={<AdminDashboard/>}/>
          <Route path="/user/filterPage"  element={<FilterPage/>}/>
          <Route path="/resultsloader" element={<ResultsLoader/>}/>
          <Route path="/resultsPage" element={<ResultsPage/>}/>
          <Route path="/user/multipleFilesUpload" element={<MultipleFilesUpload/>}/>
         </Routes>
        
      </div>
    </Provider>
  );
}

export default App;
