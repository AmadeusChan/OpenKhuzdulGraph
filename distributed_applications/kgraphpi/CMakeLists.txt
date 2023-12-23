function (add_distributed_graphpi_application application_name)
	add_executable(DISTRIBUTED_GRAPHPI_${application_name} ${application_name}.cc)
	target_link_libraries(DISTRIBUTED_GRAPHPI_${application_name} ${dependencies})
	set_target_properties(DISTRIBUTED_GRAPHPI_${application_name} PROPERTIES OUTPUT_NAME ${application_name})
endfunction()

# clique
add_distributed_graphpi_application(triangle)
add_distributed_graphpi_application(four_clique)
add_distributed_graphpi_application(five_clique)

# motif
add_distributed_graphpi_application(three_motif)
#add_distributed_graphpi_application(four_motif)
#
## other patterns
#add_distributed_graphpi_application(four_chain)
#add_distributed_graphpi_application(three_star)
