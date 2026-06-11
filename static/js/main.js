const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setupNavbar() {
  const navbar = document.getElementById("navbar");
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");

  if (!navbar || !navToggle || !navLinks) {
    return;
  }

  const setScrolled = () => {
    navbar.classList.toggle("scrolled", window.scrollY > 40);
  };

  const closeMenu = () => {
    navToggle.classList.remove("active");
    navToggle.setAttribute("aria-expanded", "false");
    navLinks.classList.remove("active");
    navbar.classList.remove("open");
    document.body.classList.remove("nav-open");
  };

  navToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("active");
    navToggle.classList.toggle("active", isOpen);
    navToggle.setAttribute("aria-expanded", String(isOpen));
    navbar.classList.toggle("open", isOpen);
    document.body.classList.toggle("nav-open", isOpen);
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeMenu);
  });

  window.addEventListener("scroll", setScrolled, { passive: true });
  setScrolled();
}

function setupSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (event) => {
      const id = anchor.getAttribute("href");
      const target = id && document.querySelector(id);

      if (!target) {
        return;
      }

      event.preventDefault();
      const top = target.getBoundingClientRect().top + window.scrollY - 70;
      window.scrollTo({ top, behavior: prefersReducedMotion ? "auto" : "smooth" });
    });
  });
}

function setupReveal() {
  const targets = document.querySelectorAll("[data-reveal]");

  if (!targets.length) {
    return;
  }

  targets.forEach((target) => {
    const delay = target.dataset.delay;
    if (delay) {
      target.style.setProperty("--reveal-delay", `${delay}ms`);
    }
  });

  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    targets.forEach((target) => target.classList.add("revealed"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        entry.target.classList.add("revealed");
        observer.unobserve(entry.target);
      });
    },
    {
      threshold: 0.12,
      rootMargin: "0px 0px -6% 0px"
    }
  );

  targets.forEach((target) => observer.observe(target));
}

function setupBibtexCopy() {
  const button = document.getElementById("copyBibtex");
  const code = document.getElementById("bibtexCode");

  if (!button || !code) {
    return;
  }

  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      button.textContent = "Copied";
      button.classList.add("copied");
      window.setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove("copied");
      }, 1800);
    } catch (error) {
      button.textContent = "Select";
      window.getSelection().selectAllChildren(code);
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1800);
    }
  });
}

setupNavbar();
setupSmoothScroll();
setupReveal();
setupBibtexCopy();
