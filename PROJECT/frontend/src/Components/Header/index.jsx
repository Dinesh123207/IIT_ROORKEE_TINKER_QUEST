import React from "react";
import "./style.css";
import { Link } from "react-router-dom";
import useStoryContext from "../../hooks/useProductContext";

export default function Index() {
  const navLinks = document.querySelectorAll('.nav-link');

function scrollToSection(e) {
  e.preventDefault();
  const targetId = this.getAttribute('href').substring(1);
  const targetSection = document.getElementById(targetId);
  if (targetSection) {
    targetSection.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
}

function highlightNavLink() {
  const sections = document.querySelectorAll('section[id]');
  sections.forEach(section => {
    const sectionTop = section.offsetTop;
    const sectionHeight = section.offsetHeight;
    const navLink = document.querySelector(`.nav-link[href="#${section.id}"]`);

    if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
      navLinks.forEach(link => link.classList.remove('active'));
      navLink.classList.add('active');
    } else {
      navLink.classList.remove('active');
    }
  });
}

navLinks.forEach(link => {
  link.addEventListener('click', scrollToSection);
});

window.addEventListener('scroll', highlightNavLink);


  let { setClickedSignUp, setModalPop, setAdminPop } = useStoryContext();
  let onClickSignUp = () => {
    setClickedSignUp(false);
    setModalPop(true);
  };

  let onClickAdmin = () => {
    setAdminPop(true);
  };
  return (
    // <div className="w-11/12 flex justify-between bg-gray-100 rounded-xl mx-auto mt-8 px-8 py-4 tracking-wide font-semibold transition-all">
    //   <div className="flex gap-8">
    //     <div className="flex gap-4">
    //       <img src="sih.png" alt="sih-logo" className="h-20" />
    //       <img src="vit.png" alt="vit-logo" className="h-20" />
    //       {/* <img src="bajaj.png" alt="bajaj-logo" className="bajaj-logo" /> */}
    //     </div>
    //     <div className="flex items-center gap-4">
    //       <Link to="/about" className="hover:text-blue-700">
    //         About
    //       </Link>
    //       <Link to="/services" className="hover:text-blue-700">
    //         Services
    //       </Link>
    //       <Link to="/contact" className="hover:text-blue-700">
    //         Contact Us
    //       </Link>
    //     </div>
    //   </div>
    //   <div className="flex ">
    //     <button
    //       className="h-fit border px-8 py-4 rounded-xl bg-primary-700 hover:bg-primary-800 text-white"
    //       onClick={onClickAdmin}
    //     >
    //       Admin Login
    //     </button>
    //     <button className="" onClick={onClickSignUp}>
    //       User Login
    //     </button>
    //   </div>
    // </div>
    <header>
      <nav class="bg-gray-50 border-gray-100 dark:border-gray-800 shadow-lg px-4  dark:bg-gray-800 rounded-xl border max-w-screen-xl mx-auto my-8">
        <div class="flex flex-wrap justify-between items-center mx-auto max-w-screen-xl">
          <div className="flex items-center">
            <img src="sih.png" class="mr-3 h-6 sm:h-20" alt="Flowbite Logo" />
            <img src="vit.png" class="mr-3 h-6 sm:h-24" alt="Flowbite Logo" />
          </div>
          <div class="flex items-center gap-2 lg:order-2">
          <button
                 onClick={onClickAdmin}
                class="inline-flex items-center justify-center px-4 lg:px-5 py-2 lg:py-2.5 text-base font-medium text-center text-gray-900 border border-gray-300 rounded-lg hover:bg-gray-100 focus:ring-4 focus:ring-gray-100 dark:text-white dark:border-gray-700 dark:hover:bg-gray-700 dark:focus:ring-gray-800"
              >
                Admin Login
              </button>
            <button
              onClick={onClickSignUp}
              class="text-white bg-primary-700 hover:bg-primary-800 focus:ring-4 focus:ring-primary-300 font-medium rounded-lg text-base px-4 lg:px-5 py-2 lg:py-2.5 mr-2 dark:bg-primary-600 dark:hover:bg-primary-700 focus:outline-none dark:focus:ring-primary-800"
            >
              User Login
            </button>
          </div>
          
          <div
            class="hidden justify-between items-center w-full lg:flex lg:w-auto lg:order-1"
            id="mobile-menu-2"
          >
            <ul class="flex flex-col mt-4 font-medium lg:flex-row lg:space-x-8 lg:mt-0 sm:text-base">
              <li>
                <a
                 href="#home" 
                  class=" nav-link block py-2 pr-4 pl-3 text-white rounded bg-primary-700 lg:bg-transparent lg:text-primary-700 lg:p-0 dark:text-white"
                  aria-current="page"
                >
                  Home
                </a>
              </li>
              <li>
                <a
                  href="#features"
                  class="nav-link block py-2 pr-4 pl-3 text-gray-700 border-b border-gray-100 hover:bg-gray-50 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 lg:p-0 dark:text-gray-400 lg:dark:hover:text-white dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent dark:border-gray-700"
                >
                  Features
                </a>
              </li>

              <li>
                <a
                  href="#contact"
                  class="nav-link block py-2 pr-4 pl-3 text-gray-700 border-b border-gray-100 hover:bg-gray-50 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 lg:p-0 dark:text-gray-400 lg:dark:hover:text-white dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent dark:border-gray-700"
                >
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </header>
  );
}
