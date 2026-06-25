// NavDP Docs – main.js
// Handles navbar scroll, mobile menu, animations

document.addEventListener('DOMContentLoaded', () => {

  // --- Navbar scroll effect ---
  const navbar = document.getElementById('navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 20);
    }, { passive: true });
  }

  // --- Mobile nav toggle ---
  const toggle = document.getElementById('navToggle');
  const navLinks = document.querySelector('.nav-links');
  if (toggle && navLinks) {
    toggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
    // close on link click
    navLinks.querySelectorAll('.nav-link, .nav-cta').forEach(link => {
      link.addEventListener('click', () => navLinks.classList.remove('open'));
    });
  }

  // --- Active nav link by current page ---
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // --- Intersection Observer for fade-in animations ---
  const observerOptions = {
    threshold: 0.12,
    rootMargin: '0px 0px -40px 0px'
  };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll(
    '.card, .task-card, .baseline-card, .qs-step, .log-entry, .chapter-card, .tool-card, .arch-layer'
  ).forEach(el => {
    el.classList.add('animate-on-scroll');
    observer.observe(el);
  });

  // --- Copy code blocks ---
  document.querySelectorAll('.code-block, pre').forEach(block => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code');
    block.style.position = 'relative';
    block.appendChild(btn);

    btn.addEventListener('click', () => {
      const code = block.querySelector('code');
      const text = code ? code.innerText : block.innerText.replace('Copy', '').trim();
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
      });
    });
  });

  // Inject copy button CSS dynamically
  const style = document.createElement('style');
  style.textContent = `
    .animate-on-scroll { opacity: 0; transform: translateY(20px); transition: opacity 0.5s ease, transform 0.5s ease; }
    .animate-on-scroll.visible { opacity: 1; transform: translateY(0); }
    .copy-btn {
      position: absolute;
      top: 0.75rem; right: 0.75rem;
      padding: 0.3rem 0.75rem;
      font-size: 0.72rem;
      font-weight: 600;
      background: var(--bg-glass);
      color: var(--text-secondary);
      border: 1px solid var(--border-card);
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      font-family: inherit;
    }
    .copy-btn:hover { background: var(--bg-card-hover); color: var(--text-primary); }
    .copy-btn.copied { background: rgba(16,185,129,0.12); color: var(--accent-green); border-color: rgba(16,185,129,0.25); }
  `;
  document.head.appendChild(style);

  // --- Theme Toggle ---
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    const icon = themeToggle.querySelector('.theme-toggle-icon');
    
    // Function to update icon based on active theme
    const updateIcon = (theme) => {
      if (icon) {
        icon.textContent = theme === 'light' ? '🌙' : '☀️';
      }
    };

    // Initialize icon based on current state
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    updateIcon(currentTheme);

    themeToggle.addEventListener('click', () => {
      const activeTheme = document.documentElement.getAttribute('data-theme') || 'light';
      const newTheme = activeTheme === 'light' ? 'dark' : 'light';
      
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      updateIcon(newTheme);
    });
  }

  // --- Smooth scroll for anchor links ---
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        const offset = 80;
        window.scrollTo({ top: target.offsetTop - offset, behavior: 'smooth' });
      }
    });
  });
});
