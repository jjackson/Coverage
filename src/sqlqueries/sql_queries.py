SQL_QUERIES = {
    "opportunity_uservisit": "SELECT opportunity_uservisit.id as visit_id, opportunity_opportunity.name as opportunity_name, user_id AS flw_id, users_user.name AS flw_name, visit_date, opportunity_id, status, form_json -> 'form' ->> 'du_name' AS du_name, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',1) AS lattitude, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',2) AS longitude, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',3) AS elevation_in_m, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',4) AS accuracy_in_m, flagged, flag_reason, form_json -> 'form' -> 'cluster_update_block' -> 'update_cluster_case' -> 'case' ->> '@user_id' AS cchq_user_owner_id\n  FROM public.opportunity_uservisit \n  LEFT JOIN opportunity_opportunity ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id  LEFT JOIN users_user ON users_user.id = opportunity_uservisit.user_id  "
                             "WHERE opportunity_opportunity.id IN (516,517,531,539);",

    "kmc_visit_query": """SELECT 
    opportunity_uservisit.id as visit_id,
    opportunity_opportunity.name as opportunity_name,
    user_id AS flw_id,
    users_user.name AS flw_name,
    visit_date,
    opportunity_id,
    status,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',1) AS latitude,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',2) AS longitude,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',3) AS elevation_in_m,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',4) AS accuracy_in_m,
    flagged,
    flag_reason,
    form_json -> 'form' -> 'cluster_update_block' -> 'update_cluster_case' -> 'case' ->> '@user_id' AS cchq_user_owner_id,
    
    -- Case ID (updated to use the working field):
    form_json -> 'form' ->> 'kmc_beneficiary_case_id' AS "case_id",
    form_json -> 'form'->'anthropometric'->>'child_weight_visit' as "child_weight_visit",
    form_json
FROM public.opportunity_uservisit 
LEFT JOIN opportunity_opportunity ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id
LEFT JOIN users_user ON users_user.id = opportunity_uservisit.user_id
WHERE opportunity_opportunity.id IN (524)"""
}
