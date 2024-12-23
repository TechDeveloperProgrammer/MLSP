document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Feature section hover effects
    const features = document.querySelectorAll('.feature');
    features.forEach(feature => {
        feature.addEventListener('mouseenter', () => {
            feature.style.transform = 'scale(1.05)';
            feature.style.transition = 'transform 0.3s ease';
        });

        feature.addEventListener('mouseleave', () => {
            feature.style.transform = 'scale(1)';
        });
    });

    // Code copy functionality
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const copyButton = document.createElement('button');
        copyButton.textContent = 'Copy';
        copyButton.classList.add('copy-btn');
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent).then(() => {
                copyButton.textContent = 'Copied!';
                setTimeout(() => {
                    copyButton.textContent = 'Copy';
                }, 2000);
            });
        });
        block.parentNode.insertBefore(copyButton, block);
    });
});
