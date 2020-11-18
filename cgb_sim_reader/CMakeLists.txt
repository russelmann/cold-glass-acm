cmake_minimum_required( VERSION 3.6 FATAL_ERROR )
set( CMAKE_VERBOSE_MAKEFILE ON )

set( CMAKE_CXX_STANDARD 14 )
#set( CMAKE_POSITION_INDEPENDENT_CODE ON )

get_filename_component( SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE )

set( EIGEN3_DIR ${SRC_PATH}/extern/eigen )
set( CEREAL_DIR ${SRC_PATH}/extern/cereal )

project( CgbSimReader )

file ( GLOB HEADERS
	${SRC_PATH}/*.h
)

file( GLOB SOURCES
	${SRC_PATH}/*.cpp
)

set ( INCLUDES
	${EIGEN3_DIR}
	${CEREAL_DIR}/include
)

set( SOURCES ${HEADERS} ${SOURCES})

include_directories(${INCLUDES})
add_executable( CgbSimReader ${SOURCES} )
