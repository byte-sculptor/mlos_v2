project(Mlos.UnitTest)

get_filename_component(MLOS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${MLOS_ROOT}/build/Mlos.Common.cmake")
include("${MLOS_ROOT}/build/Mlos.Cpp.cmake")
include("${MLOS_ROOT}/build/Mlos.Cpp.UnitTest.cmake")

add_executable(${PROJECT_NAME}
    BufferTests.cpp
    FileWatchEventTests.Linux.cpp
    Main.cpp
    MessageVerificationTests.cpp
    MetadataTests.cpp
    PlatformTests.cpp
    SharedChannelTests.cpp
    UniqueStringTests.cpp)

target_link_libraries(${PROJECT_NAME} Mlos.Core)

#find_package(GTest REQUIRED)
target_link_libraries(${PROJECT_NAME} gtest)

# Add dependencies on SettingsRegistry project(s).

# Mlos.NetCore currently also does the Mlos.Core SettingsProvider code generation.
add_dependencies(${PROJECT_NAME} Mlos.NetCore)

add_subdirectory(Mlos.UnitTest.SettingsRegistry)
add_dependencies(${PROJECT_NAME} Mlos.UnitTest.SettingsRegistry)

# Use a custom function to help setup a test run.
add_mlos_agent_server_exe_test_run(
    NAME MlosTestRun_Mlos.Agent.Server_${PROJECT_NAME}
    EXECUTABLE_TARGET ${PROJECT_NAME}
    TIMEOUT 600)

# Binplace the outputs for more convenient testing.
install(TARGETS ${PROJECT_NAME}
    CONFIGURATIONS ${CMAKE_BUILD_TYPE}
    RUNTIME DESTINATION ${BINPLACE_DIR})
