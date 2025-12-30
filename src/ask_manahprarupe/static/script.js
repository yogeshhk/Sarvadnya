function highlightAnswers() {
    const answers = document.querySelectorAll('.answer-item');
    answers.forEach(answer => {
        answer.addEventListener('mouseenter', () => {
            answer.style.backgroundColor = '#f0f8ff';
        });
        answer.addEventListener('mouseleave', () => {
            answer.style.backgroundColor = 'white';
        });
    });
}

document.addEventListener('DOMContentLoaded', highlightAnswers);
