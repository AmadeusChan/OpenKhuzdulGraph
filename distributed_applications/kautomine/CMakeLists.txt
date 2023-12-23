function (add_distributed_automine_application application_name)
	add_executable(DISTRIBUTED_AUTOMINE_${application_name} ${application_name}.cc)
	target_link_libraries(DISTRIBUTED_AUTOMINE_${application_name} ${dependencies})
	set_target_properties(DISTRIBUTED_AUTOMINE_${application_name} PROPERTIES OUTPUT_NAME ${application_name})
endfunction()

# cliques
add_distributed_automine_application(triangle)
add_distributed_automine_application(four_clique)
add_distributed_automine_application(five_clique)

add_distributed_automine_application(three_motif)

# chains
#add_distributed_automine_application(edge_induced_three_chain)

# FSM
add_distributed_automine_application(three_fsm)
