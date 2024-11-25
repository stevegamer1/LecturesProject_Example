# Add the source files in this directory
set(SOURCES src/ConvNet.cu)

# Add these sources to the project executable
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})