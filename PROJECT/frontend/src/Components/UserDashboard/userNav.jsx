import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function UserNav() {
  let navigate = useNavigate(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const profileShow = isDropdownOpen ? "block" : "hidden";

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  let onClickLogout = () => {
    localStorage.clear();
    navigate("/");
    navigate(0);
  }

  return (
    <div className="flex absolute top-0 left-0 bg-primary-900 border-b border-gray-600 shadow-2xl z-30 w-full h-20 text-white">
      <nav className="w-full">
        <div class="flex  h-full flex-wrap justify-between items-center mx-auto w-full">
          {/* <div className="flex items-center">
            <img src="sih.png" class="mr-3 h-6 sm:h-20" alt="Flowbite Logo" />
            <img src="vit.png" class="mr-3 h-6 sm:h-24" alt="Flowbite Logo" />
          </div> */}
          <div class="flex gap-12 justify-between px-8  h-full items-center md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
            <button
              type="button"
              className="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600"
              id="user-menu-button"
              aria-expanded={isDropdownOpen ? "true" : "false"}
              // onClick={toggleDropdown}
              aria-controls="user-dropdown"
            >
              <img
                class="w-12 h-30 rounded-full bg-white gap-16"
                src="/notification.svg"
                
                alt="notification photo"
              />

            {/* <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" id="notifications"
            >
                <path d="M34,20H32a8,8,0,0,0-8-8V10A10,10,0,0,1,34,20Zm7,17a3,3,0,0,1-3,3H30.92a7,7,0,0,1-13.84,0H10a3,3,0,0,1,0-6V20A14,14,0,0,1,23,6.05V2h2V6.05A14,14,0,0,1,38,20V34A3,3,0,0,1,41,37ZM12,34H36V20a12,12,0,0,0-24,0Zm16.9,6H19.1a5,5,0,0,0,9.8,0ZM39,37a1,1,0,0,0-1-1H10a1,1,0,0,0,0,2H38A1,1,0,0,0,39,37Z" data-name="59 Notifications, Alarm, Notification">
    </path></svg> */}
            </button>
            <button
              type="button"
              className="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600"
              id="user-menu-button"
              aria-expanded={isDropdownOpen ? "true" : "false"}
              onClick={toggleDropdown}
              aria-controls="user-dropdown"
            >
              {/* <button type="button" class="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600" id="user-menu-button" aria-expanded="false" data-dropdown-toggle="user-dropdown" data-dropdown-placement="bottom"> */}
              <span class="sr-only">Open user menu</span>
              <img
                class="w-12 h-30 rounded-full "
                src="/user_profile.png"
                alt="user photo"
              />
              {/* <!-- Dropdown menu --> */}
              <div
                class={`${profileShow} z-50 fixed top-10 xl:right-8 my-4 text-base list-none bg-white divide-y divide-gray-100 rounded-lg shadow dark:bg-gray-700 dark:divide-gray-600`}
                id="user-dropdown"
              >
                <div class="px-4 py-3">
                  <span class="block text-sm text-gray-900 dark:text-white">
                    UserName
                  </span>
                  <span class="block text-sm  text-gray-500 truncate dark:text-gray-400">
                    user123@gmail.com
                  </span>
                </div>
                <ul class="py-2" aria-labelledby="user-menu-button">
                  <li>
                    <a
                      href="#f"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
                    >
                      Dashboard
                    </a>
                  </li>
                  {/* < li>
                    <a
                      href="#f"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
                    >
                      My Account
                    </a>
                     </li> */}
                  <li>
                    <a
                      href="#f"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
                    >
                      Update Contact Details
                    </a>
                  </li>

                  <li>
                    <a
                      href="#f"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
                    >
                      View Payments Report
                    </a>
                  </li>
                  <li>
                    <a
                      href="#f"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
                    >
                      View Insurance Details
                    </a>
                  </li>
                  <li>
                    <a
                      href="#f"
                      onClick={onClickLogout}
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
                    >
                      Logout
                    </a>
                  </li>
                </ul>
              </div>
            </button>

            <button
              data-collapse-toggle="navbar-user"
              type="button"
              class="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-gray-500 rounded-lg md:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600"
              aria-controls="navbar-user"
              aria-expanded="false"
            >
              <span class="sr-only">Open main menu</span>
              <svg
                class="w-5 h-5"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 17 14"
              >
                <path
                  stroke="currentColor"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M1 1h15M1 7h15M1 13h15"
                />
              </svg>
            </button>
          </div>

          <div
            class="hidden justify-between items-center w-full lg:flex lg:w-auto lg:order-1"
            id="mobile-menu-2"
          ></div>
        </div>
      </nav>
    </div>
  );
}





// import React, { useState } from "react";

// export default function UserNav() {
//   const [isDropdownOpen, setIsDropdownOpen] = useState(false);
//   const profileShow = isDropdownOpen ? "block" : "hidden";

//   const toggleDropdown = () => {
//     setIsDropdownOpen(!isDropdownOpen);
//   };

//   return (
//     <div className="flex items-center  sticky top-0 left-0 bg-primary-900 border-b border-gray-600 shadow-2xl z-50 w-full h-20 text-white">
//       <nav className="w-full">
//         <div class="flex flex-wrap justify-between items-center mx-auto max-w-screen-xl">
//           {/* <div className="flex items-center">
//             <img src="sih.png" class="mr-3 h-6 sm:h-20" alt="Flowbite Logo" />
//             <img src="vit.png" class="mr-3 h-6 sm:h-24" alt="Flowbite Logo" />
//           </div> */}
//           <div class="flex items-center md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
//             <button
//               type="button"
//               className="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600"
//               id="user-menu-button"
//               aria-expanded={isDropdownOpen ? "true" : "false"}
//               onClick={toggleDropdown}
//               aria-controls="user-dropdown"
//             >
//               <img
//                 class="w-12 h-30 rounded-full "
//                 src="notification.png"
//                 alt="user photo"
//               ></img>

//             </button>
//             <button
//               type="button"
//               className="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600"
//               id="user-menu-button"
//               aria-expanded={isDropdownOpen ? "true" : "false"}
//               onClick={toggleDropdown}
//               aria-controls="user-dropdown"
//             >
//               {/* <button type="button" class="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600" id="user-menu-button" aria-expanded="false" data-dropdown-toggle="user-dropdown" data-dropdown-placement="bottom"> */}
//               <span class="sr-only">Open user menu</span>
//               <img
//                 class="w-12 h-30 rounded-full "
//                 src="user_profile.png"
//                 alt="user photo"
//               ></img>
//               {/* <!-- Dropdown menu --> */}
//               <div
//                 class={`${profileShow} z-50 fixed top-10 my-4 text-base list-none bg-white divide-y divide-gray-100 rounded-lg shadow dark:bg-gray-700 dark:divide-gray-600`}
//                 id="user-dropdown"
//               >
//                 <div class="px-4 py-3">
//                   <span class="block text-sm text-gray-900 dark:text-white">
//                     UserName 
//                   </span>
//                   <span class="block text-sm  text-gray-500 truncate dark:text-gray-400">
//                     user123@gmail.com
//                   </span>
//                 </div>
//                 <ul class="py-2" aria-labelledby="user-menu-button">
//                   <li>
//                     <a
//                       href="#"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       Dashboard
//                     </a>
//                   </li>
//                   {/* < li>
//                     <a
//                       href="#"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       My Account
//                     </a>
//                      </li> */}
//                    <li>
//                     <a
//                       href="#"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       Update Contact Details 
//                     </a>
//                   </li>

//                   <li>
//                     <a
//                       href="#"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       View Payments Report 
//                     </a>
//                   </li>
//                   <li>
//                     <a
//                       href="#"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       View Insurance Details 
//                     </a>
//                   </li>
//                   <li>
//                     <a
//                       href="#"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       Logout
//                     </a>
//                   </li>
//                 </ul>
//               </div>
//             </button>

//             <button
//               data-collapse-toggle="navbar-user"
//               type="button"
//               class="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-gray-500 rounded-lg md:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600"
//               aria-controls="navbar-user"
//               aria-expanded="false"
//             >
//               <span class="sr-only">Open main menu</span>
//               <svg
//                 class="w-5 h-5"
//                 aria-hidden="true"
//                 xmlns="http://www.w3.org/2000/svg"
//                 fill="none"
//                 viewBox="0 0 17 14"
//               >
//                 <path
//                   stroke="currentColor"
//                   stroke-linecap="round"
//                   stroke-linejoin="round"
//                   stroke-width="2"
//                   d="M1 1h15M1 7h15M1 13h15"
//                 />
//               </svg>
//             </button>
//           </div>

//           <div
//             class="hidden justify-between items-center w-full lg:flex lg:w-auto lg:order-1"
//             id="mobile-menu-2"
//           >
//             <ul class="flex items-center mt-4 font-medium lg:flex-row lg:space-x-8 lg:mt-0 sm:text-base">
//               <li className="px-6 py-3 bg-transparent hover:bg-primary-600 rounded transition-colors">
//                 <a
//                   href="#f"
//                   class="block py-2 pl-2 pr-3 text-white border-b border-gray-100 hover:bg-gray-50 lg:hover:bg-transparent lg:border-0 lg:hover:text-gray-200 lg:p-0 dark:text-gray-400 lg:dark:hover:text-white dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent dark:border-gray-700"
//                 >
//                   Home
//                 </a>
//               </li>
//               <li className="px-6 py-3 bg-transparent hover:bg-primary-600 rounded transition-colors">
//                 <a
//                   href="#f"
//                   class="block py-2 pl-2 pr-3 text-white border-b border-gray-100 hover:bg-gray-50 lg:hover:bg-transparent lg:border-0 lg:hover:text-gray-200 lg:p-0 dark:text-gray-400 lg:dark:hover:text-white dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent dark:border-gray-700"
//                 >
//                   Features
//                 </a>
//               </li>
//               <li className="px-6 py-3 bg-transparent hover:bg-primary-600 rounded transition-colors">
//                 <a
//                   href="#f"
//                   class="block py-2 pl-2 pr-3 text-white border-b border-gray-100 hover:bg-gray-50 lg:hover:bg-transparent lg:border-0 lg:hover:text-gray-200 lg:p-0 dark:text-gray-400 lg:dark:hover:text-white dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent dark:border-gray-700"
//                 >
//                   Contact
//                 </a>
//               </li>
//             </ul>
//           </div>

        
//         </div>
//       </nav>
//     </div>
//   );
// }



// import React, { useState } from "react";

// export default function UserNav() {
//   const [isDropdownOpen, setIsDropdownOpen] = useState(false);
//   const profileShow = isDropdownOpen ? "block" : "hidden";

//   const toggleDropdown = () => {
//     setIsDropdownOpen(!isDropdownOpen);
//   };

//   return (
//     <div className="flex absolute top-0 left-0 bg-primary-900 border-b border-gray-600 shadow-2xl z-30 w-full h-20 text-white">
//       <nav className="w-full">
//         <div class="flex  h-full flex-wrap justify-between items-center mx-auto max-w-screen-xl">
//           {/* <div className="flex items-center">
//             <img src="sih.png" class="mr-3 h-6 sm:h-20" alt="Flowbite Logo" />
//             <img src="vit.png" class="mr-3 h-6 sm:h-24" alt="Flowbite Logo" />
//           </div> */}
//           <div class="flex gap-4 justify-between px-8  h-full  items-center md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
//             <button
//               type="button"
//               className="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600"
//               id="user-menu-button"
//               aria-expanded={isDropdownOpen ? "true" : "false"}
//               onClick={toggleDropdown}
//               aria-controls="user-dropdown"
//             >
//               <img
//                 class="w-12 h-30 rounded-full "
//                 src="notification.png"
//                 alt="user photo"
//               />
//             </button>
//             <button
//               type="button"
//               className="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600"
//               id="user-menu-button"
//               aria-expanded={isDropdownOpen ? "true" : "false"}
//               onClick={toggleDropdown}
//               aria-controls="user-dropdown"
//             >
//               {/* <button type="button" class="flex text-sm bg-gray-800 rounded-full md:me-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600" id="user-menu-button" aria-expanded="false" data-dropdown-toggle="user-dropdown" data-dropdown-placement="bottom"> */}
//               <span class="sr-only">Open user menu</span>
//               <img
//                 class="w-12 h-30 rounded-full "
//                 src="user_profile.png"
//                 alt="user photo"
//               />
//               {/* <!-- Dropdown menu --> */}
//               <div
//                 class={`${profileShow} z-50 fixed top-10 xl:right-8 my-4 text-base list-none bg-white divide-y divide-gray-100 rounded-lg shadow dark:bg-gray-700 dark:divide-gray-600`}
//                 id="user-dropdown"
//               >
//                 <div class="px-4 py-3">
//                   <span class="block text-sm text-gray-900 dark:text-white">
//                     UserName
//                   </span>
//                   <span class="block text-sm  text-gray-500 truncate dark:text-gray-400">
//                     user123@gmail.com
//                   </span>
//                 </div>
//                 <ul class="py-2" aria-labelledby="user-menu-button">
//                   <li>
//                     <a
//                       href="#f"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       Dashboard
//                     </a>
//                   </li>
//                   {/* < li>
//                     <a
//                       href="#f"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       My Account
//                     </a>
//                      </li> */}
//                   <li>
//                     <a
//                       href="#f"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       Update Contact Details
//                     </a>
//                   </li>

//                   <li>
//                     <a
//                       href="#f"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       View Payments Report
//                     </a>
//                   </li>
//                   <li>
//                     <a
//                       href="#f"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       View Insurance Details
//                     </a>
//                   </li>
//                   <li>
//                     <a
//                       href="#f"
//                       class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
//                     >
//                       Logout
//                     </a>
//                   </li>
//                 </ul>
//               </div>
//             </button>

//             <button
//               data-collapse-toggle="navbar-user"
//               type="button"
//               class="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-gray-500 rounded-lg md:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600"
//               aria-controls="navbar-user"
//               aria-expanded="false"
//             >
//               <span class="sr-only">Open main menu</span>
//               <svg
//                 class="w-5 h-5"
//                 aria-hidden="true"
//                 xmlns="http://www.w3.org/2000/svg"
//                 fill="none"
//                 viewBox="0 0 17 14"
//               >
//                 <path
//                   stroke="currentColor"
//                   stroke-linecap="round"
//                   stroke-linejoin="round"
//                   stroke-width="2"
//                   d="M1 1h15M1 7h15M1 13h15"
//                 />
//               </svg>
//             </button>
//           </div>

//           <div
//             class="hidden justify-between items-center w-full lg:flex lg:w-auto lg:order-1"
//             id="mobile-menu-2"
//           ></div>
//         </div>
//       </nav>
//     </div>
//   );
// }
