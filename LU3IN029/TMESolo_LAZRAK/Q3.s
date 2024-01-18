.data
str:	.asciiz	"abcba"
ok:	.asciiz	"oui"
ko:	.asciiz	"non"


.text

main:

# prologue
	addiu	$29, $29, -8 #2 variable locale+0 appel pile
	

	
# n=len(str)	
	lui	$8,0x1001
	ori	$4,$8,0
	jal 	len
	sw	$2,0($29)



# res=est_palin
	lui	$8,0x1001
# str
	ori	$4,$8,0
# n-1
	lw	$5,0($29)
	addiu	$5,$5,-1
	jal 	est_palin
	ori	$9,$2,0	
	lui	$8,0x1001
# if (res==1)
	blez	$9, else
	addiu	$8,$8,6
# printf(%s,ok)
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
	j	endif
else:		
# printf(%s,ko)
	addiu	$8,$8,10
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
endif:
#epilogue
	addiu	$29,$29,8

# exit 
	ori	$2,$0,10
	syscall
	
	
	
	
	
len:
# prologue
	addiu	$29,$29,-8 # 1 variable locale+$31
	sw	$31,4($29)
# c=0
	ori	$2,$0,0
	sw	$2,0($29)
	
	
while:
	lb	$10,0($4)
	beq	$10,$0, endwhile
	addiu	$2,$2,1
	addiu	$4,$4,1
	j while

endwhile:
	
# epilogue
	addiu	$29,$29,8
	jr	$31
	
	
	
	
	
est_palin:
# prologue
	addiu	$29,$29,-20	# 2 arg+2 variable locale + $31
	sw	$31,16($29)
# corps de fontction

#int idx_dep=0
	ori	$8,$0,0
	
	# str[fin]
	addu	$11,$4,$5 
while1:
# while (idx_deb<=idx_fin)

	sub	$10,$8,$5
	beq	$10,$0,endwhile1
# str[debut]
	lb	$13,0($4)

	lb	$12,0($11)

#if	str[debut]!=str[fin]
	bne	$13,$11, endwhile0
	addiu	$11,$11,-1
	addiu	$4,$4,1
	j while1

endwhile0:
	ori	$2,$0,0
# epilogue
	addiu	$29,$29,20
	jr $31

endwhile1:
	ori	$2,$0,1
# epilogue
	addiu	$29,$29,20
	jr $31
