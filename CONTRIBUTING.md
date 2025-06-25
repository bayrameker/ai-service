# Contributing to AI Service

Thank you for your interest in contributing to the AI Service project! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Use the appropriate issue template**:
   - Bug Report: For reporting bugs
   - Feature Request: For suggesting new features
   - Question: For asking questions

3. **Provide detailed information**:
   - Clear description of the issue or feature
   - Steps to reproduce (for bugs)
   - Environment details
   - Expected vs actual behavior

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation if needed

4. **Test your changes**:
   ```bash
   # Run tests
   ./backend/scripts/run_tests.py
   
   # Run specific test categories
   ./backend/scripts/run_tests.py --unit
   ./backend/scripts/run_tests.py --integration
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   - Use a clear title and description
   - Reference any related issues
   - Ensure all tests pass

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bayrameker/ai-service.git
   cd ai-service
   ```

2. **Run setup script**:
   ```bash
   ./scripts/setup.sh
   ```

3. **Add your API keys**:
   ```bash
   # Edit backend/.env file
   nano backend/.env
   ```

4. **Start development server**:
   ```bash
   ./scripts/deploy.sh dev
   ```

## Code Style Guidelines

- **Python**: Follow PEP 8 style guidelines
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Testing**: Write tests for all new functionality
- **Logging**: Use appropriate logging levels

## Testing

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test system performance under load
- **Coverage**: Maintain at least 80% test coverage

## Documentation

- Update API documentation for new endpoints
- Add docstrings to all new functions and classes
- Update README.md if needed
- Add examples for new features

## Community Guidelines

- **Be respectful**: Treat all community members with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Maintainers review contributions in their spare time
- **Follow the code of conduct**: Maintain a welcoming environment

## Questions?

If you have questions about contributing:

1. Check existing **GitHub Discussions**
2. Create a new **Discussion** for general questions
3. Create an **Issue** using the Question template
4. Review the documentation at `/docs` endpoint

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Project documentation

Thank you for helping make AI Service better! 🚀
