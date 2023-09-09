$(document).ready(function(){
    $(window).scroll(function(){
        if(this.scrollY > 20){
            $('.navbar').addClass(".sticky");
        }else{
            $('.navbar').removeClass("sticky");
        }
        if(this.scrollY > 500){
           $('.scroll-up-btn').addClass("show"); 
        }else{
            $('.scroll-up-btn').removeClass("show");
        }
    });
    
    //slide-up script
        $('.scroll-up-btn').click(function(){
            $('html').animate({scrollTop: 0});
        });

    // toggle menu / navbar script

    $('.menu-btn').click(function(){
        $('.navbar .menu').toggleClass("active");
        $('.menu-btn i').toggleClass("active");
    });


    //typing text animation
    var typed = new Typed(".typing", {
        strings: ["Web Developer" , "Dsa Learner", "Qt - UI Designer"],
        typeSpeed: 100,
        backSpeed: 60,
        loop: true
    });

    var typed = new Typed(".typing-2", {
        strings: ["Web Developer" , "Dsa Learner", "Qt - UI Designer"],
        typeSpeed: 100,
        backSpeed: 60,
        loop: true
    });


    // ------------------------------------------- Qualification --------------------------------------------

    //owl caraousel 
    
    $('.carousel').owlCarousel({
        margin: 20,
        loop: true,
        autoplayTimeOut: 2000,
        autoplayHoverPause: true,
        responsive: {
            0:{
                items: 1,
                nav: false
            },
            650:{
                items:2,
                nav:false
            },
            1065:{
                items: 3,
                nav: false
            }
        }

    });
});



 
  /*===============qualification tabs=======================*/
  
  const tabs = document.querySelectorAll('[data-target]'),
     tabContents = document.querySelectorAll('[data-content]')
  
  
     tabs.forEach(tab=>{
        tab.addEventListener('click', () =>{
         const target = document.querySelector(tab.dataset.target)
         tabContents.forEach(tabContent => {
            tabContent.classList.remove('qualification__active')
         }) 
         target.classList.add('qualification__active')
  
         tabs.forEach(tab =>{
            tab.classList.remove('qualification__active')
         })
         tab.classList.add('qualification__active')
        })
     })
  


    /******************************************portfolio academice based projects**************************************/
    
     /*=============Projects===========================*/
     const skillsContent = document.getElementsByClassName('skills__content'),
     skillsHeader=document.querySelectorAll('.skills__header')
     
     function toggleSkills(){
      let itemClass = this.parentNode.className
     
      for(i=0; i < skillsContent.length; i++)
      {
          skillsContent[i].className = 'skills__content skills__close'
      }
      if(itemClass === 'skills__content skills__close'){
          this.parentNode.className = 'skills__content skills__open'
      }
     }
     skillsHeader.forEach((el) => {
       el.addEventListener('click' , toggleSkills)
     })
     
     /********************************scroll*********************** */
     var swiper = new Swiper(".mySwiper", {
        slidesPerView: 3,
        spaceBetween: 30,
        slidesPerGroup: 3,
        loop: true,
        loopFillGroupWithBlank: true,
        pagination: {
          el: ".swiper-pagination",
          clickable: true,
        },
        navigation: {
          nextEl: ".swiper-button-next",
          prevEl: ".swiper-button-prev",
        },
      });  
     
      var swiper = new Swiper(".slide-content", {
       slidesPerView: 3,
       spaceBetween: 25,
       loop: true,
       centerSlide: 'true',
       fade: 'true',
       grabCursor: 'true',
       pagination: {
         el: ".swiper-pagination",
         clickable: true,
         dynamicBullets: true,
       },
       navigation: {
         nextEl: ".swiper-button-next",
         prevEl: ".swiper-button-prev",
       },
     
       breakpoints:{
           0: {
               slidesPerView: 1,
           },
           520: {
               slidesPerView: 2,
           },
           950: {
               slidesPerView: 3,
           },
       },
     });
     
  var swiper = new Swiper(".slide-content", {
      slidesPerView: 3,
      spaceBetween: 25,
      loop: true,
      centerSlide: 'true',
      fade: 'true',
      grabCursor: 'true',
      pagination: {
        el: ".swiper-pagination",
        clickable: true,
        dynamicBullets: true,
      },
      navigation: {
        nextEl: ".swiper-button-next",
        prevEl: ".swiper-button-prev",
      },
  
      breakpoints:{
          0: {
              slidesPerView: 1,
          },
          520: {
              slidesPerView: 2,
          },
          950: {
              slidesPerView: 3,
          },
      },
    });

